#include "app/cli.hpp"

#include "app/presets.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <format>
#include <print>
#include <set>

namespace mako::app {

namespace {

const char* env(const std::string& name)
{
    const char* v = std::getenv(name.c_str());
    return v && *v ? v : nullptr;
}

} // namespace

Cli::Cli(std::string tool, std::string positionals, size_t min_positionals, size_t max_positionals)
    : tool_(std::move(tool)), positionals_(std::move(positionals)), min_positionals_(min_positionals),
      max_positionals_(max_positionals)
{
}

Cli& Cli::option(std::string name, std::string value_name, std::string help,
                 std::function<void(const std::string&)> set)
{
    std::string group = name;
    opts_.push_back({std::move(name), std::move(value_name), std::move(help), std::move(group), std::move(set), {}});
    return *this;
}

Cli& Cli::flag(std::string name, std::string help, std::function<void(bool)> set)
{
    std::string group = name;
    opts_.push_back({std::move(name), "", std::move(help), std::move(group), {}, std::move(set)});
    return *this;
}

Cli& Cli::shared_env(std::string_view other)
{
    for (const auto& o : opts_)
        if (o.name == other) {
            opts_.back().group = o.group;
            return *this;
        }
    throw std::logic_error(std::format("shared_env: no option {}", other));
}

Cli& Cli::secret()
{
    opts_.back().secret = true;
    return *this;
}

std::string Cli::env_name(std::string_view option)
{
    if (!option.starts_with("--"))
        return {};
    std::string name = "MAKO_";
    for (const char c : option.substr(2))
        name += c == '-' ? '_' : char(std::toupper(static_cast<unsigned char>(c)));
    return name;
}

int64_t Cli::parse_int(const std::string& v)
{
    int64_t x = 0;
    const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), x);
    if (ec == std::errc::result_out_of_range)
        throw std::invalid_argument("is out of range");
    if (ec != std::errc() || end != v.data() + v.size() || v.empty())
        throw std::invalid_argument("takes an integer");
    return x;
}

uint64_t Cli::parse_uint(const std::string& v)
{
    uint64_t x = 0;
    const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), x);
    if (ec == std::errc::result_out_of_range)
        throw std::invalid_argument("is out of range");
    if (ec != std::errc() || end != v.data() + v.size() || v.empty())
        throw std::invalid_argument("takes a non-negative integer");
    return x;
}

double Cli::parse_double(const std::string& v)
{
    char* end = nullptr;
    errno = 0;
    const double x = v.empty() || std::isspace(static_cast<unsigned char>(v[0])) ? NAN : std::strtod(v.c_str(), &end);
    if (end != v.c_str() + v.size() || !std::isfinite(x) || errno == ERANGE)
        throw std::invalid_argument("takes a number");
    return x;
}

std::pair<size_t, bool> Cli::find(std::string_view arg) const
{
    for (size_t i = 0; i < opts_.size(); ++i) {
        const auto& o = opts_[i];
        if (arg == o.name)
            return {i, false};
        if (o.is_flag() && arg.starts_with("--no-") && arg.substr(5) == std::string_view(o.name).substr(2))
            return {i, true};
    }
    return {opts_.size(), false};
}

std::string Cli::usage() const
{
    return std::format("usage: {} {} [options] (--help for the options)", tool_, positionals_);
}

std::string Cli::help() const
{
    std::string out = std::format("usage: {} {} [options]\n\nOptions (each also from the environment variable after it; "
                                  "the command line wins):\n",
                                  tool_, positionals_);
    for (const auto& o : opts_) {
        std::string spec = o.is_flag() ? std::format("--[no-]{}", o.name.substr(2)) : o.name + " " + o.value_name;
        std::string line = std::format("  {:<30}", spec);
        if (spec.size() > 30)
            line += "\n" + std::string(32, ' ');
        line += o.help;
        if (const auto e = env_name(o.name); !e.empty())
            line += std::format("{}({})", o.help.empty() ? "" : " ", e);
        out += line + "\n";
    }
    out += "  --preset NAME                 options from a preset (MAKO_PRESET)\n";
    out += std::format("  --presets FILE                the presets file, default {} (MAKO_PRESETS)\n",
                       presets_path().string());
    return out;
}

std::vector<std::string> Cli::parse(int argc, char** argv)
{
    try {
        return parse(std::vector<std::string>(argv + std::min(argc, 1), argv + argc));
    } catch (const Help& h) {
        std::print("{}", h.text);
        std::exit(0);
    } catch (const std::exception& e) {
        std::println(stderr, "{}: {}\n{}", tool_, e.what(), usage());
        std::exit(2);
    }
}

std::vector<std::string> Cli::parse(std::vector<std::string> in)
{
    const auto fail = [](const std::string& why) { throw std::runtime_error(why); };
    in.insert(in.begin(), tool_);
    for (size_t i = 1; i < in.size(); ++i)
        if (in[i] == "--help" || in[i] == "-h")
            throw Help{help()};

    // MAKO_PRESETS and MAKO_PRESET unless the command line names its own; the preset goes before the first option
    // (after leading positionals, so that a preset's model fills them only when the command line gives none).
    const auto has = [&](std::string_view a) { return std::ranges::find(in, a) != in.end(); };
    if (const char* f = env("MAKO_PRESETS"); f && !has("--presets")) {
        in.push_back("--presets");
        in.push_back(f);
        std::println(stderr, "{}: --presets {} from MAKO_PRESETS", tool_, f);
    }
    if (const char* p = env("MAKO_PRESET"); p && !has("--preset")) {
        auto at = std::find_if(in.begin() + 1, in.end(), [](const std::string& a) { return a.starts_with("-"); });
        at = in.insert(at, p);
        in.insert(at, "--preset");
        std::println(stderr, "{}: --preset {} from MAKO_PRESET", tool_, p);
    }

    std::vector<char*> ptrs;
    for (auto& a : in)
        ptrs.push_back(a.data());
    ptrs.push_back(nullptr);
    Args expanded(int(in.size()), ptrs.data());
    const std::vector<std::string> args(expanded.argv(), expanded.argv() + expanded.argc());

    std::vector<std::string> positionals;
    std::vector<Use> uses;
    std::set<std::string> given; // groups the command line gives
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (!a.starts_with("-") || a == "-") {
            positionals.push_back(a);
            continue;
        }
        const auto [k, negated] = find(a);
        if (k == opts_.size())
            fail(std::format("unknown option {}", a));
        given.insert(opts_[k].group);
        if (opts_[k].is_flag()) {
            uses.push_back({k, negated ? "0" : "1", {}});
            continue;
        }
        if (i + 1 >= args.size())
            fail(std::format("option {} needs a value ({})", a, opts_[k].value_name));
        uses.push_back({k, args[++i], {}});
    }
    if (positionals.size() < min_positionals_ || positionals.size() > max_positionals_)
        fail(positionals.size() < min_positionals_ ? "missing arguments" : std::format("unexpected argument {}", positionals.back()));

    std::vector<Use> from_env;
    for (size_t k = 0; k < opts_.size(); ++k) {
        const auto name = env_name(opts_[k].name);
        const char* v = name.empty() || given.contains(opts_[k].group) ? nullptr : env(name);
        if (!v)
            continue;
        std::string value = v;
        if (opts_[k].is_flag()) {
            const std::string_view s(v);
            if (s == "1" || s == "true" || s == "yes" || s == "on")
                value = "1";
            else if (s == "0" || s == "false" || s == "no" || s == "off")
                value = "0";
            else
                fail(std::format("{}={} takes 1 or 0", name, s));
            std::println(stderr, "{}: {}{} from {}", tool_, value == "1" ? "--" : "--no-", opts_[k].name.substr(2), name);
        } else {
            std::println(stderr, "{}: {} {} from {}", tool_, opts_[k].name, opts_[k].secret ? "(secret)" : value, name);
        }
        from_env.push_back({k, std::move(value), name});
    }
    uses.insert(uses.begin(), from_env.begin(), from_env.end());

    for (const auto& u : uses) {
        const auto& o = opts_[u.opt];
        try {
            if (o.is_flag())
                o.set_flag(u.value == "1");
            else
                o.set_value(u.value);
        } catch (const std::exception& e) {
            fail(std::format("{} {} (got {}{})", o.name, e.what(), o.secret ? "(secret)" : u.value,
                             u.env.empty() ? "" : " from " + u.env));
        }
    }
    return positionals;
}

} // namespace mako::app
