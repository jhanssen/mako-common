#include "memory/borrower.hpp"

#include "log/log.hpp"
#include "memory/lender.hpp"

#include <glaze/glaze.hpp>
#include <httplib.h>

#include <format>
#include <stdexcept>

namespace mako::memory {

Borrower::Borrower(std::string url, std::string api_key, std::string name)
    : url_(std::move(url)), key_(std::move(api_key)), name_(std::move(name))
{
}

Borrower::~Borrower()
{
    release();
}

void Borrower::acquire()
{
    std::unique_lock lock(m_);
    cv_.wait(lock, [&] { return state_ != State::waiting; });
    if (state_ == State::held)
        return;
    if (thread_.joinable()) { // a connection that ended: its thread is done
        lock.unlock();
        thread_.join();
        lock.lock();
    }
    state_ = State::waiting;
    asked_ = false;
    error_.clear();
    client_ = std::make_unique<httplib::Client>(url_);
    client_->set_connection_timeout(std::chrono::seconds(10));
    client_->set_read_timeout(std::chrono::hours(24)); // the lender may wait long for its own work to end
    if (!key_.empty())
        client_->set_bearer_token_auth(key_);
    thread_ = std::jthread([this] {
        httplib::Request req;
        req.method = "POST";
        req.path = kAcquirePath;
        std::string name;
        (void)glz::write_json(name_, name);
        req.body = std::format(R"({{"name":{}}})", name);
        req.set_header("Content-Type", "application/json");
        int status = 0;
        std::string pending, err_body;
        req.response_handler = [&](const httplib::Response& r) {
            status = r.status;
            return true;
        };
        req.content_receiver = [&](const char* data, size_t n, uint64_t, uint64_t) {
            if (status != 200) {
                err_body.append(data, n);
                return true;
            }
            pending.append(data, n);
            for (size_t end; (end = pending.find("\n\n")) != std::string::npos; pending.erase(0, end + 2)) {
                const std::string_view block(pending.data(), end);
                if (block.starts_with("event: granted")) {
                    {
                        std::lock_guard g(m_);
                        state_ = State::held;
                    }
                    cv_.notify_all();
                } else if (block.starts_with("event: release")) {
                    {
                        std::lock_guard g(m_);
                        asked_ = true;
                    }
                    if (on_release)
                        on_release();
                }
            }
            return true;
        };
        httplib::Response res;
        httplib::Error error = httplib::Error::Success;
        client_->send(req, res, error);
        bool lost = false;
        {
            std::lock_guard g(m_);
            if (state_ == State::waiting) {
                state_ = State::failed;
                error_ = status && status != 200
                             ? std::format("{}: HTTP {}: {}", url_, status, err_body)
                             : std::format("{}: {}", url_, httplib::to_string(error));
            } else if (state_ == State::held) { // ended without release(): the lender stopped
                lost = !asked_;
                asked_ = true;
                state_ = State::idle;
            }
        }
        cv_.notify_all();
        if (lost) {
            log::warn("memory: the connection to {} ended", url_);
            if (on_release)
                on_release();
        }
    });
    cv_.wait(lock, [&] { return state_ != State::waiting; });
    if (state_ == State::failed) {
        state_ = State::idle;
        const std::string e = error_;
        lock.unlock();
        thread_.join();
        throw std::runtime_error(std::format("borrowing memory: {}", e));
    }
}

void Borrower::release()
{
    {
        std::lock_guard lock(m_);
        if (state_ == State::held)
            state_ = State::idle; // the thread sees it was given back
    }
    if (client_)
        client_->stop();
    if (thread_.joinable() && thread_.get_id() != std::this_thread::get_id())
        thread_.join();
    cv_.notify_all();
}

bool Borrower::held()
{
    std::lock_guard lock(m_);
    return state_ == State::held;
}

bool Borrower::asked()
{
    std::lock_guard lock(m_);
    return state_ == State::held && asked_;
}

} // namespace mako::memory
