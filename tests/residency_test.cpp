// The model in memory (memory/residency.hpp) with a stand-in model and a lender on loopback: loading borrows the
// LLM's memory, a reclaim frees the model once its lease ends, the next lease borrows again, another key replaces the
// model, a failed load gives the memory back, and without a lender nothing is borrowed.
//   residency_test
#include "memory/borrower.hpp"
#include "memory/lender.hpp"
#include "memory/residency.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <format>
#include <functional>
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

std::atomic<int> alive = 0; // stand-in models in memory

struct Model {
    std::string key;
    explicit Model(std::string k) : key(std::move(k)) { ++alive; }
    ~Model() { --alive; }
};

bool eventually(const std::function<bool()>& f)
{
    for (int i = 0; i < 200; ++i) {
        if (f())
            return true;
        std::this_thread::sleep_for(25ms);
    }
    return f();
}

} // namespace

int main()
{
    std::atomic<int> lent = 0, returned = 0;
    mako::memory::Lender lender({.lend = [&](const std::string&) { ++lent; }, .returned = [&] { ++returned; }});
    httplib::Server server;
    lender.install(server);
    const int port = server.bind_to_any_port("127.0.0.1");
    std::thread t([&] { server.listen_after_bind(); });
    server.wait_until_ready();

    using Residency = mako::memory::Residency<Model>;
    int loads = 0;
    const auto load = [&](const std::string& key) {
        return [&loads, key] {
            ++loads;
            return std::make_unique<Model>(key);
        };
    };
    {
        Residency r(std::make_unique<mako::memory::Borrower>(std::format("http://127.0.0.1:{}", port), "", "test"));
        check(std::string(r.state()) == "none" && lent == 0, "nothing loaded or borrowed before a lease");
        {
            auto m = r.acquire("a", "model a", load("a"));
            check(m->key == "a" && lent == 1 && lender.lent() && loads == 1 && r.loaded() == "model a",
                  "the first lease borrows the LLM's memory, then loads");
        }
        {
            auto m = r.acquire("a", "model a", load("a"));
            check(loads == 1 && lent == 1, "the same key again: loaded already, nothing borrowed");
            lender.reclaim();
            std::this_thread::sleep_for(300ms);
            check(alive == 1 && returned == 0, "a reclaim during a lease waits for it to end");
        }
        check(eventually([&] { return returned == 1 && alive == 0; }) && std::string(r.state()) == "none",
              "after the lease: the model freed and the memory returned");
        {
            auto m = r.acquire("a", "model a", load("a"));
            check(lent == 2 && loads == 2 && alive == 1, "the next lease borrows and loads again");
        }
        {
            auto m = r.acquire("b", "model b", load("b"));
            check(m->key == "b" && loads == 3 && alive == 1 && lent == 2, "another key replaces the model");
        }
        lender.reclaim();
        check(eventually([&] { return returned == 2 && alive == 0; }), "an idle model is freed at once on a reclaim");
        bool threw = false;
        try {
            (void)r.acquire("c", "model c", [] () -> std::unique_ptr<Model> { throw std::runtime_error("bad files"); });
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw && eventually([&] { return returned == 3; }) && !lender.lent(),
              "a failed load throws and gives the memory back");
        {
            auto m = r.acquire("a", "model a", load("a"));
            check(lent == 4 && alive == 1, "a lease after a failed one borrows again");
        }
    }
    check(eventually([&] { return returned == 4 && alive == 0; }), "the residency going away frees and returns");

    {
        Residency r(nullptr);
        auto m = r.acquire("a", "model a", load("a"));
        check(alive == 1 && lent == 4, "without a lender: loads, borrows nothing");
    }
    lender.stop();
    server.stop();
    t.join();
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
