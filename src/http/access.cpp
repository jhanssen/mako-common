#include "http/access.hpp"

#include <string_view>

namespace mako::http {

bool key_matches(const std::string& authorization, const std::string& x_api_key, const std::vector<std::string>& keys)
{
    const auto same = [](std::string_view got, const std::string& key) {
        if (key.empty() || got.size() != key.size())
            return false;
        unsigned char diff = 0;
        for (size_t i = 0; i < key.size(); ++i)
            diff |= static_cast<unsigned char>(got[i] ^ key[i]);
        return diff == 0;
    };
    constexpr std::string_view bearer = "Bearer ";
    const std::string_view b = authorization.starts_with(bearer) ? std::string_view(authorization).substr(bearer.size())
                                                                  : std::string_view();
    bool ok = false;
    for (const auto& key : keys) { // every key compared, whichever matches
        ok |= same(b, key);
        ok |= same(x_api_key, key);
    }
    return ok;
}

namespace {

// The host part of "name[:port]" or "[v6]:port".
std::string_view host_part(std::string_view h)
{
    if (h.starts_with('['))
        return h.substr(0, h.find(']') + 1);
    return h.substr(0, h.find(':'));
}

bool loopback_name(std::string_view h)
{
    return h == "localhost" || h == "127.0.0.1" || h == "[::1]";
}

} // namespace

bool host_allowed(const std::string& host)
{
    return host.empty() || loopback_name(host_part(host));
}

bool origin_allowed(const std::string& origin)
{
    if (origin.empty())
        return true;
    std::string_view o(origin);
    for (const std::string_view scheme : {"http://", "https://"})
        if (o.starts_with(scheme))
            return loopback_name(host_part(o.substr(scheme.size())));
    return false;
}

bool allowed(const std::string& authorization, const std::string& x_api_key, const std::string& host,
             const std::string& origin, const std::vector<std::string>& keys)
{
    return keys.empty() ? host_allowed(host) && origin_allowed(origin) : key_matches(authorization, x_api_key, keys);
}

} // namespace mako::http
