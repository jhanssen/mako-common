#include "gguf/read_pool.hpp"

#include <condition_variable>
#include <cerrno>
#include <cstring>
#include <exception>
#include <format>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <vector>

#include <unistd.h>

namespace mako::gguf {

void read_exact(int fd, void* dst, uint64_t bytes, uint64_t offset)
{
    auto* p = static_cast<char*>(dst);
    while (bytes) {
        const ssize_t n = ::pread(fd, p, bytes, static_cast<off_t>(offset));
        if (n <= 0)
            throw std::runtime_error(std::format("pread at {}: {}", offset, n < 0 ? std::strerror(errno) : "EOF"));
        p += n;
        bytes -= static_cast<uint64_t>(n);
        offset += static_cast<uint64_t>(n);
    }
}

struct ReadPool::Impl {
    explicit Impl(int threads)
    {
        for (int t = 0; t < threads; ++t)
            threads_.emplace_back([this](std::stop_token st) { work(st); });
    }
    ~Impl()
    {
        for (auto& t : threads_)
            t.request_stop();
        cv_.notify_all();
    }

    void run(size_t n, const std::function<void(size_t)>& task)
    {
        std::unique_lock lock(m_);
        task_ = &task;
        n_ = n;
        next_ = 0;
        done_ = 0;
        error_ = nullptr;
        ++generation_;
        cv_.notify_all();
        done_cv_.wait(lock, [&] { return done_ == n_; });
        task_ = nullptr;
        if (error_)
            std::rethrow_exception(error_);
    }

    void work(std::stop_token st)
    {
        uint64_t seen = 0;
        std::unique_lock lock(m_);
        for (;;) {
            cv_.wait(lock, st, [&] { return generation_ != seen; });
            if (st.stop_requested())
                return;
            seen = generation_;
            while (task_ && next_ < n_) {
                const size_t i = next_++;
                lock.unlock();
                std::exception_ptr err;
                try {
                    (*task_)(i);
                } catch (...) {
                    err = std::current_exception();
                }
                lock.lock();
                if (err && !error_)
                    error_ = err;
                if (++done_ == n_)
                    done_cv_.notify_one();
            }
        }
    }

    std::mutex m_;
    std::condition_variable_any cv_, done_cv_;
    const std::function<void(size_t)>* task_ = nullptr;
    size_t n_ = 0, next_ = 0, done_ = 0;
    uint64_t generation_ = 0;
    std::exception_ptr error_;
    std::vector<std::jthread> threads_;
};

ReadPool::ReadPool(int threads) : impl_(std::make_unique<Impl>(threads)) {}
ReadPool::~ReadPool() = default;

void ReadPool::run(size_t n, const std::function<void(size_t)>& task)
{
    impl_->run(n, task);
}

} // namespace mako::gguf
