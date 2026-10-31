// HTTP access checks: API keys, and without keys the Host and Origin headers.
//   http_test
#include "http/access.hpp"

#include <print>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

} // namespace

int main()
{
    using namespace mako::http;
    const std::vector<std::string> keys{"s3cret-one-0123456", "s3cret-two-0123456"};
    check(key_matches("Bearer s3cret-two-0123456", "", keys), "an API key as a bearer token (the second of two)");
    check(key_matches("", "s3cret-one-0123456", keys), "an API key as x-api-key");
    check(!key_matches("Bearer s3cret-one-012345", "", keys) && !key_matches("Bearer s3cret-one-01234567", "", keys),
          "a shorter or longer key is refused");
    check(!key_matches("s3cret-one-0123456", "", keys), "the Authorization header needs the Bearer scheme");
    check(!key_matches("Bearer wrong", "wrong", keys) && !key_matches("", "", keys), "a wrong or missing key is refused");
    check(!key_matches("Bearer ", "", {""}) && !key_matches("Bearer x", "x", {}), "no or empty keys match nothing");
    check(host_allowed("localhost:8080") && host_allowed("127.0.0.1") && host_allowed("[::1]:8080") && host_allowed(""),
          "loopback Host headers (and none)");
    check(!host_allowed("evil.example.com") && !host_allowed("localhost.evil.com:8080") &&
              !host_allowed("192.168.1.5:8080"),
          "other Host headers refused without a key");
    check(origin_allowed("") && origin_allowed("http://localhost:3000") && origin_allowed("https://127.0.0.1") &&
              origin_allowed("http://[::1]:8080"),
          "loopback Origins (and none)");
    check(!origin_allowed("https://evil.example.com") && !origin_allowed("http://localhost.evil.com") &&
              !origin_allowed("null") && !origin_allowed("file://"),
          "other Origins refused without a key");
    check(allowed("", "", "localhost:8080", "", {}) && !allowed("", "", "evil.example.com", "", {}) &&
              allowed("Bearer s3cret-one-0123456", "", "evil.example.com", "https://evil.example.com", keys) &&
              !allowed("", "", "localhost", "", keys),
          "allowed: keys when there are some, else loopback");
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
