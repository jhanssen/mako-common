#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace mako::app {

// A tool's command-line options, read (lowest precedence first) from the environment, presets and the command line:
//  - Each long option --some-name has the variable MAKO_SOME_NAME (short options like -n have none). A variable
//    takes the option's value; a flag's takes 1, true, yes or on (the flag) or 0, false, no or off (--no-flag).
//    MAKO_PRESETS and MAKO_PRESET stand for --presets and --preset. Empty variables are ignored.
//  - An option the command line (or a preset it names) gives ignores its variable, and so do the options sharing
//    a group with it (shared_env), as options that add up must (--api-key, --api-key-file).
//  - Otherwise options apply in order, the last one winning: those from the environment first, then the command
//    line with each --preset NAME in place (app::Args).
//  - A flag --name has --no-name to clear it. Positional arguments may stand anywhere among the options.
// Options taken from the environment are listed on stderr; --help lists every option and its variable.
class Cli {
public:
    // tool: the program's name in messages; positionals: their names for the usage line ("<model.gguf> <id,...>"),
    // between min and max of them.
    Cli(std::string tool, std::string positionals, size_t min_positionals, size_t max_positionals);

    // An option taking a value, handed to `set`; `set` throws (std::invalid_argument) with what it takes.
    Cli& option(std::string name, std::string value_name, std::string help, std::function<void(const std::string&)> set);

    // An option taking a value read into `to`: a string, an integer, a floating-point number or a bool (0 or 1).
    template <typename T>
    Cli& option(std::string name, std::string value_name, std::string help, T& to)
    {
        return option(std::move(name), std::move(value_name), std::move(help),
                      [&to](const std::string& v) { to = parse<T>(v); });
    }

    // A flag: --name gives true, --no-name false.
    Cli& flag(std::string name, std::string help, std::function<void(bool)> set);
    Cli& flag(std::string name, std::string help, bool& to)
    {
        return flag(std::move(name), std::move(help), [&to](bool b) { to = b; });
    }

    // The option declared last shares `other`'s group: the command line giving either ignores both variables.
    Cli& shared_env(std::string_view other);
    // The option declared last is a secret: messages leave its value out.
    Cli& secret();

    // The positionals, or (printing why on stderr) exit(2) on an error and exit(0) after printing --help.
    std::vector<std::string> parse(int argc, char** argv);
    // The same from the arguments after the program name; throws std::runtime_error (why) on an error and Help
    // (what --help prints) on --help.
    struct Help {
        std::string text;
    };
    std::vector<std::string> parse(std::vector<std::string> args);

    // The variable of a long option (--arena-gb: MAKO_ARENA_GB), empty for a short one.
    static std::string env_name(std::string_view option);

    template <typename T> static T parse(const std::string& v)
    {
        if constexpr (std::is_same_v<T, std::string>) {
            return v;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (v == "1" || v == "true")
                return true;
            if (v == "0" || v == "false")
                return false;
            throw std::invalid_argument("takes 0 or 1");
        } else if constexpr (std::integral<T> && std::is_signed_v<T>) {
            const int64_t x = parse_int(v);
            if (x < int64_t(std::numeric_limits<T>::min()) || x > int64_t(std::numeric_limits<T>::max()))
                throw std::invalid_argument("is out of range");
            return T(x);
        } else if constexpr (std::integral<T>) {
            const uint64_t x = parse_uint(v);
            if (x > uint64_t(std::numeric_limits<T>::max()))
                throw std::invalid_argument("is out of range");
            return T(x);
        } else {
            static_assert(std::floating_point<T>, "an option reads into a string, an integer, a float or a bool");
            return T(parse_double(v));
        }
    }

private:
    struct Opt {
        std::string name, value_name, help, group;
        std::function<void(const std::string&)> set_value;
        std::function<void(bool)> set_flag;
        bool secret = false;
        bool is_flag() const { return bool(set_flag); }
    };
    // One option occurrence: its index in opts_, the value (a flag's "1" or "0") and where it came from.
    struct Use {
        size_t opt;
        std::string value;
        std::string env; // the variable it came from, empty from the command line
    };

    static int64_t parse_int(const std::string& v);
    static uint64_t parse_uint(const std::string& v);
    static double parse_double(const std::string& v);

    // The option `arg` names and, for a flag, whether it is the --no- form; opts_.size() when none.
    std::pair<size_t, bool> find(std::string_view arg) const;
    std::string usage() const;
    std::string help() const;

    std::string tool_, positionals_;
    size_t min_positionals_, max_positionals_;
    std::vector<Opt> opts_;
};

} // namespace mako::app
