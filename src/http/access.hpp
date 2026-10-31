#pragma once

#include <string>
#include <vector>

namespace mako::http {

// Whether a request's credentials (the Authorization or x-api-key header value) carry one of `keys`; each compared in
// constant time.
bool key_matches(const std::string& authorization, const std::string& x_api_key, const std::vector<std::string>& keys);
// Without keys: whether a Host header names this machine's loopback (localhost, 127.0.0.1, [::1], with any port) or is
// empty (a client that sends none).
bool host_allowed(const std::string& host);
// Without keys: whether an Origin header is absent or a loopback page (http(s)://localhost, 127.0.0.1 or [::1]).
bool origin_allowed(const std::string& origin);
// Whether a request may in: with keys, one of them; without, from this machine's loopback and no other web page.
bool allowed(const std::string& authorization, const std::string& x_api_key, const std::string& host,
             const std::string& origin, const std::vector<std::string>& keys);

} // namespace mako::http
