#pragma once

#include "log/log.hpp"
#include "memory/borrower.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace mako::memory {

// A model in memory, used by one lease at a time (a generation server's). A lease names what it needs by a key (another key than the
// loaded one's replaces it). With a borrower (an LLM server on this machine lends its memory), loading first borrows
// the LLM's memory; when the LLM wants it back, the model is freed and the memory returned as soon as no lease is
// out, and the next lease borrows again.
template <typename E> class Residency {
public:
    using Load = std::function<std::unique_ptr<E>()>;
    // borrower: may be null (the model stays loaded).
    explicit Residency(std::unique_ptr<memory::Borrower> borrower) : borrower_(std::move(borrower))
    {
        if (!borrower_)
            return;
        borrower_->on_release = [this] {
            {
                std::lock_guard lock(m_);
                give_back_ = true;
            }
            cv_.notify_all();
        };
        returner_ = std::jthread([this] {
            std::unique_lock lock(m_);
            for (;;) {
                cv_.wait(lock, [&] { return stopping_ || (give_back_ && !out_); });
                if (stopping_)
                    return;
                if (model_)
                    log::info("the LLM wants its memory back: unloading {}", name_);
                unload_locked();
                lock.unlock();
                borrower_->release();
                lock.lock();
                give_back_ = false;
                cv_.notify_all();
            }
        });
    }
    ~Residency()
    {
        {
            std::lock_guard lock(m_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (returner_.joinable())
            returner_.join();
        model_.reset();
        if (borrower_)
            borrower_->release();
    }
    Residency(const Residency&) = delete;
    Residency& operator=(const Residency&) = delete;

    class Lease {
    public:
        Lease(Residency& r, E& e) : r_(&r), e_(&e) {}
        Lease(Lease&& o) noexcept : r_(std::exchange(o.r_, nullptr)), e_(o.e_) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease()
        {
            if (r_)
                r_->end_lease();
        }
        E& operator*() const { return *e_; }
        E* operator->() const { return e_; }

    private:
        Residency* r_;
        E* e_;
    };

    // Waits for the lease before it (and for memory the LLM asked back to be returned), then loads `key` with `load`
    // unless it is loaded; name: for the log. Throws what loading (or borrowing) throws.
    Lease acquire(const std::string& key, const std::string& name, const Load& load)
    {
        std::unique_lock lock(m_);
        cv_.wait(lock, [&] { return !out_ && !give_back_; });
        out_ = true;
        struct Out {
            Residency& r;
            std::unique_lock<std::mutex>& lock;
            bool keep = false;
            ~Out()
            {
                if (keep)
                    return;
                if (!lock.owns_lock())
                    lock.lock();
                r.out_ = false;
                r.loading_ = false;
                r.cv_.notify_all();
            }
        } out{*this, lock};
        if (model_ && key_ != key) {
            log::info("{} asked for: loading it", name == name_ ? "other LoRAs" : name);
            unload_locked();
        }
        if (!model_) {
            loading_ = true;
            lock.unlock(); // state() and loaded() answer meanwhile; out_ keeps other leases out
            const auto t0 = std::chrono::steady_clock::now();
            if (borrower_ && !borrower_->held()) {
                log::info("borrowing the LLM's memory");
                borrower_->acquire();
                log::info("the LLM's memory is free ({:.1f} s)", secs_since(t0));
            }
            std::unique_ptr<E> m;
            try {
                m = load();
            } catch (...) {
                if (borrower_) // nothing loaded: the LLM may have its memory back
                    borrower_->release();
                throw;
            }
            lock.lock();
            loading_ = false;
            model_ = std::move(m);
            key_ = key;
            name_ = name;
            log::info("{} loaded in {:.1f} s", name, secs_since(t0));
        }
        out.keep = true;
        return Lease(*this, *model_);
    }

    // The name of what is loaded, "" if nothing.
    std::string loaded()
    {
        std::lock_guard lock(m_);
        return name_;
    }
    // "loaded", "none", or "loading" (borrowing memory or reading weights).
    const char* state()
    {
        std::lock_guard lock(m_);
        return loading_ ? "loading" : model_ ? "loaded" : "none";
    }

private:
    static double secs_since(std::chrono::steady_clock::time_point t)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    }
    void end_lease()
    {
        {
            std::lock_guard lock(m_);
            out_ = false;
        }
        cv_.notify_all();
    }
    void unload_locked()
    {
        model_.reset();
        key_.clear();
        name_.clear();
    }

    std::unique_ptr<memory::Borrower> borrower_;
    std::mutex m_;
    std::condition_variable cv_;
    std::unique_ptr<E> model_;
    std::string key_, name_;
    bool out_ = false, loading_ = false;
    bool give_back_ = false; // the lender asked for its memory
    bool stopping_ = false;
    std::jthread returner_; // frees the model and returns the memory when asked and no lease is out
};

} // namespace mako::memory
