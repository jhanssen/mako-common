#include "memory/lender.hpp"

#include "log/log.hpp"

#include <glaze/glaze.hpp>
#include <httplib.h>

#include <chrono>
#include <format>

namespace mako::memory {

namespace {

constexpr auto kKeepalive = std::chrono::seconds(5);

std::string error_json(const std::string& message)
{
    std::string m;
    (void)glz::write_json(message, m);
    return std::format(R"({{"error":{{"message":{},"type":"server_error"}}}})", m);
}

} // namespace

void Lender::install(httplib::Server& server)
{
    server.Post(kAcquirePath, [this](const httplib::Request& req, httplib::Response& res) {
        std::string name = "a borrower";
        glz::generic body;
        if (!req.body.empty() && !glz::read_json(body, req.body) && body.is_object() && body.contains("name") &&
            body["name"].is_string())
            name = body["name"].get_string();
        {
            std::unique_lock lock(m_);
            cv_.wait(lock, [&] { return !lent_ || stopping_; });
            if (stopping_) {
                res.status = 503;
                res.set_content(error_json("the server is stopping"), "application/json");
                return;
            }
            lent_ = true;
            want_back_ = false;
        }
        try {
            hooks_.lend(name);
        } catch (const std::exception& e) {
            {
                std::lock_guard lock(m_);
                lent_ = false;
            }
            cv_.notify_all();
            res.status = 503;
            res.set_content(error_json(e.what()), "application/json");
            return;
        }
        log::info("memory lent to {}", name);
        const auto closed = req.is_connection_closed;
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, closed](size_t, httplib::DataSink& sink) {
                const auto send = [&](std::string_view s) { return sink.write(s.data(), s.size()); };
                if (!send("event: granted\ndata: {}\n\n"))
                    return false;
                auto last = std::chrono::steady_clock::now();
                bool asked = false;
                std::unique_lock lock(m_);
                while (!closed() && !stopping_) {
                    cv_.wait_for(lock, std::chrono::seconds(1), [&] { return (want_back_ && !asked) || stopping_; });
                    if (stopping_)
                        break;
                    std::string out;
                    if (want_back_ && !asked) {
                        asked = true;
                        out = "event: release\ndata: {}\n\n";
                    } else if (std::chrono::steady_clock::now() - last >= kKeepalive) {
                        out = ": keepalive\n\n";
                    }
                    if (out.empty())
                        continue;
                    lock.unlock();
                    const bool ok = send(out);
                    lock.lock();
                    last = std::chrono::steady_clock::now();
                    if (!ok)
                        break;
                }
                return false; // the borrower closed: the loan is over
            },
            [this](bool) { give_back(); }); // always called, however the response ends
    });
}

void Lender::give_back()
{
    hooks_.returned();
    {
        std::lock_guard lock(m_);
        lent_ = false;
        want_back_ = false;
    }
    cv_.notify_all();
    log::info("memory returned");
}

void Lender::reclaim()
{
    {
        std::lock_guard lock(m_);
        if (!lent_ || want_back_)
            return;
        want_back_ = true;
    }
    cv_.notify_all();
}

void Lender::stop()
{
    {
        std::lock_guard lock(m_);
        stopping_ = true;
    }
    cv_.notify_all();
}

bool Lender::lent()
{
    std::lock_guard lock(m_);
    return lent_;
}

} // namespace mako::memory
