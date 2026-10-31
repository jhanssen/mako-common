// Lending memory between processes (memory/lender.hpp), both ends in this process over loopback HTTP: a grant, a
// release asked for and given, a refusal, one loan at a time, and a borrower told when the lender goes away.
//   memory_test
#include "memory/borrower.hpp"
#include "memory/lender.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <format>
#include <mutex>
#include <print>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

// A flag one thread sets and another waits for.
struct Event {
    std::mutex m;
    std::condition_variable cv;
    int count = 0;
    void set()
    {
        {
            std::lock_guard l(m);
            ++count;
        }
        cv.notify_all();
    }
    bool wait(int n, std::chrono::milliseconds t = 10s)
    {
        std::unique_lock l(m);
        return cv.wait_for(l, t, [&] { return count >= n; });
    }
};

} // namespace

int main()
{
    Event lent, returned;
    std::atomic<bool> refuse = false;
    std::string last_name;
    mako::memory::Lender lender({.lend =
                                     [&](const std::string& name) {
                                         if (refuse)
                                             throw std::runtime_error("busy");
                                         last_name = name;
                                         lent.set();
                                     },
                                 .returned = [&] { returned.set(); }});
    httplib::Server server;
    lender.install(server);
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread t([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    const std::string url = std::format("http://127.0.0.1:{}", port);

    {
        mako::memory::Borrower b(url, "", "images");
        Event asked;
        b.on_release = [&] { asked.set(); };
        b.acquire();
        check(b.held() && lent.count == 1 && lender.lent() && last_name == "images",
              "acquire: the lender frees its memory and grants it");
        b.acquire();
        check(lent.count == 1, "acquire while held: nothing more");
        lender.reclaim();
        check(asked.wait(1) && b.asked(), "reclaim: the borrower is asked to release");
        check(returned.count == 0, "nothing is returned before the borrower releases");
        b.release();
        check(returned.wait(1) && !b.held(), "release: the lender has its memory back");
        b.acquire();
        check(lent.wait(2) && b.held() && !b.asked(), "acquire again after a release");
    } // the destructor gives it back
    check(returned.wait(2), "a borrower that goes away gives the memory back");

    refuse = true;
    {
        mako::memory::Borrower b(url, "", "refused");
        bool threw = false;
        try {
            b.acquire();
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("busy") != std::string::npos;
        }
        check(threw && !b.held() && !lender.lent(), "a refused loan: acquire throws with the lender's reason");
    }
    refuse = false;

    {
        mako::memory::Borrower a(url, "", "first"), b(url, "", "second");
        a.acquire();
        std::atomic<bool> b_held = false;
        std::thread tb([&] {
            b.acquire();
            b_held = true;
        });
        std::this_thread::sleep_for(500ms);
        check(!b_held, "one loan at a time: a second borrower waits");
        a.release();
        tb.join();
        check(b_held && b.held() && last_name == "second", "the second borrower gets it once the first returns it");
        b.release();
        check(returned.wait(4), "and returns it");
    }

    {
        mako::memory::Borrower b(url, "", "orphan");
        Event asked;
        b.on_release = [&] { asked.set(); };
        b.acquire();
        lender.stop();
        server.stop();
        t.join();
        check(asked.wait(1) && !b.held(), "the lender going away: the borrower is told to free its memory");
    }

    {
        mako::memory::Borrower b(url, "", "nobody");
        bool threw = false;
        try {
            b.acquire();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "no lender: acquire throws");
    }
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
