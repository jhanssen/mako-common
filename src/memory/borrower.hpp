#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace httplib {
class Client;
}

namespace mako::memory {

// The borrowing side of the protocol in memory/lender.hpp: borrows a lender's memory before loading, gives it back on
// request (or when done).
class Borrower {
public:
    // url: the lender's base URL (http://host:port); api_key: sent as a bearer token when not empty; name: how the
    // lender's log calls this process.
    Borrower(std::string url, std::string api_key, std::string name);
    ~Borrower(); // gives back what it holds
    Borrower(const Borrower&) = delete;
    Borrower& operator=(const Borrower&) = delete;

    // Called (on the connection's thread) when the lender asks for its memory back. Set before acquire().
    std::function<void()> on_release;

    // Returns once the memory is granted (at once if held). Throws std::runtime_error when the lender refuses or
    // cannot be reached.
    void acquire();
    // Gives the memory back; the caller has freed its own. Returns once the connection is closed.
    void release();
    bool held();
    // The lender has asked for its memory back (and it is still held).
    bool asked();

private:
    void finish();

    std::string url_, key_, name_;
    std::mutex m_;
    std::condition_variable cv_;
    std::unique_ptr<httplib::Client> client_;
    std::jthread thread_;
    enum class State { idle, waiting, held, failed } state_ = State::idle;
    bool asked_ = false;
    std::string error_;
};

} // namespace mako::memory
