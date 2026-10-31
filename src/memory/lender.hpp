#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>

namespace httplib {
class Server;
}

// Lending GPU memory between two processes on one machine: a lender (the process that holds the memory, e.g. an LLM
// server) frees it for a borrower (e.g. an image generator) and gets it back when it needs it again.
//
//   POST /v1/memory/acquire {"name": "..."}  ->  text/event-stream
//     event: granted    the lender's memory is free: the borrower may load
//     event: release    the lender wants it back: the borrower ends what it is doing, frees its memory and closes
//     ": keepalive" comments in between
//   The borrower gives the memory back by closing the connection (so also by exiting). One loan at a time: another
//   acquire waits until the one before is returned. Refused (before granted): an HTTP error status with a JSON body.
namespace mako::memory {

inline constexpr char kAcquirePath[] = "/v1/memory/acquire";

class Lender {
public:
    struct Hooks {
        // Frees the lender's memory; returns once it is free. Throws to refuse the loan.
        std::function<void(const std::string& borrower)> lend;
        // The borrower closed the connection: its memory is free again.
        std::function<void()> returned;
    };
    explicit Lender(Hooks hooks) : hooks_(std::move(hooks)) {}
    Lender(const Lender&) = delete;
    Lender& operator=(const Lender&) = delete;

    // Adds the acquire endpoint (access checks are the server's).
    void install(httplib::Server& server);
    // Asks the borrower for the memory back; returns at once (Hooks::returned says when it is back). Nothing when
    // nothing is lent.
    void reclaim();
    bool lent();
    // Ends any loan (the borrower is told the connection ended) and refuses new ones; before the server stops, which
    // waits for open connections.
    void stop();

private:
    void give_back();

    Hooks hooks_;
    std::mutex m_;
    std::condition_variable cv_;
    bool lent_ = false, want_back_ = false, stopping_ = false;
};

} // namespace mako::memory
