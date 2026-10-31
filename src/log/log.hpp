#pragma once

#include <chrono>
#include <cstdio>
#include <ctime>
#include <format>
#include <mutex>
#include <print>
#include <string>
#include <utility>

// Timestamped lines on stderr ("[2026-10-25 21:50:12.345] message"), whole lines even from several threads. For what a
// running program reports; command-line errors and usage text stay plain.
namespace mako::log {

inline std::mutex& mutex()
{
    static std::mutex m;
    return m;
}

inline std::string stamp()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm lt{};
    localtime_r(&t, &lt);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday,
                       lt.tm_hour, lt.tm_min, lt.tm_sec, ms);
}

inline void line(std::string_view level, const std::string& message)
{
    const std::string s = stamp();
    std::lock_guard lock(mutex());
    std::println(stderr, "[{}] {}{}", s, level, message);
}

template <typename... A>
void info(std::format_string<A...> f, A&&... a)
{
    line("", std::format(f, std::forward<A>(a)...));
}

template <typename... A>
void warn(std::format_string<A...> f, A&&... a)
{
    line("warning: ", std::format(f, std::forward<A>(a)...));
}

template <typename... A>
void error(std::format_string<A...> f, A&&... a)
{
    line("error: ", std::format(f, std::forward<A>(a)...));
}

} // namespace mako::log
