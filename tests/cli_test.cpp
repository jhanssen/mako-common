// The tools' option parsing (app::Cli): values, flags, the environment (MAKO_*), presets and their precedence, and
// the errors.
//   cli_test <scratch dir>
#include "app/cli.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

// The options of a made-up tool (names unlikely to be set in the environment running the test).
struct Tool {
    std::string text = "unset";
    int number = 0;
    uint32_t count = 0;
    double ratio = 0;
    bool on = false, sw = true;
    std::vector<std::string> keys;
    int short_n = 0;

    mako::app::Cli cli()
    {
        mako::app::Cli c("zztool", "<model>", 1, 2);
        c.option("--zz-text", "S", "a string", text)
            .option("--zz-number", "N", "an int", number)
            .option("--zz-count", "N", "an unsigned", count)
            .option("--zz-ratio", "F", "a double", ratio)
            .option("--zz-on", "0|1", "a bool value", on)
            .flag("--zz-switch", "a flag, default on", sw)
            .option("--zz-key", "K", "adds a key", [this](const std::string& v) { keys.push_back(v); })
            .secret()
            .option("--zz-key-file", "F", "adds a key from a name", [this](const std::string& v) { keys.push_back("file:" + v); })
            .shared_env("--zz-key")
            .option("-n", "N", "a short option", short_n);
        return c;
    }
};

// Parses with the given environment (variables set for the call, unset after).
struct Run {
    Tool t;
    std::vector<std::string> positionals;
    std::string error;
    bool help = false;
};

Run run(std::vector<std::string> args, std::vector<std::pair<std::string, std::string>> env = {})
{
    for (const auto& [k, v] : env)
        setenv(k.c_str(), v.c_str(), 1);
    Run r;
    try {
        r.positionals = r.t.cli().parse(args);
    } catch (const mako::app::Cli::Help& h) {
        r.help = true;
        r.error = h.text;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    for (const auto& [k, v] : env)
        unsetenv(k.c_str());
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    const std::filesystem::path dir = argc > 1 ? argv[1] : std::filesystem::temp_directory_path();
    for (const char* v : {"MAKO_PRESET", "MAKO_PRESETS"})
        unsetenv(v);

    std::println("values");
    {
        auto r = run({"m", "--zz-text", "hi", "--zz-number", "-5", "--zz-count", "7", "--zz-ratio", "0.25", "--zz-on",
                      "1", "-n", "3"});
        check(r.error.empty() && r.positionals == std::vector<std::string>{"m"} && r.t.text == "hi" &&
                  r.t.number == -5 && r.t.count == 7 && r.t.ratio == 0.25 && r.t.on && r.t.short_n == 3,
              "every type read");
        r = run({"--zz-count", "4", "m", "--zz-count", "9", "second"});
        check(r.t.count == 9 && r.positionals == std::vector<std::string>{"m", "second"},
              "the last occurrence wins; positionals between options");
        check(run({"m", "--zz-count", "-1"}).error == "--zz-count takes a non-negative integer (got -1)",
              "negative unsigned rejected");
        check(run({"m", "--zz-count", "12x"}).error == "--zz-count takes a non-negative integer (got 12x)",
              "trailing text rejected");
        check(run({"m", "--zz-count", "5000000000"}).error == "--zz-count is out of range (got 5000000000)",
              "out of range rejected");
        check(run({"m", "--zz-ratio", "x"}).error == "--zz-ratio takes a number (got x)", "bad number rejected");
        check(run({"m", "--zz-ratio", "inf"}).error == "--zz-ratio takes a number (got inf)", "infinity rejected");
        check(run({"m", "--zz-on", "yes"}).error == "--zz-on takes 0 or 1 (got yes)", "bad bool rejected");
    }

    std::println("flags");
    {
        check(run({"m"}).t.sw, "default kept");
        check(!run({"m", "--no-zz-switch"}).t.sw, "--no- form clears");
        check(run({"m", "--no-zz-switch", "--zz-switch"}).t.sw, "the last form wins");
    }

    std::println("errors and help");
    {
        check(run({"m", "--zz-bogus"}).error == "unknown option --zz-bogus", "unknown option");
        check(run({"m", "--zz-count"}).error == "option --zz-count needs a value (N)", "missing value");
        check(run({}).error == "missing arguments", "too few positionals");
        check(run({"a", "b", "c"}).error == "unexpected argument c", "too many positionals");
        check(run({"m", "--zz-key", "s3cret", "--zz-count", "x"}).error.find("s3cret") == std::string::npos,
              "a secret stays out of other errors");
        const auto h = run({"--help"});
        check(h.help && h.error.find("--zz-ratio F") != std::string::npos &&
                  h.error.find("(MAKO_ZZ_RATIO)") != std::string::npos &&
                  h.error.find("--[no-]zz-switch") != std::string::npos && h.error.find("MAKO_N") == std::string::npos,
              "--help lists options and variables (none for -n)");
    }

    std::println("environment");
    {
        auto r = run({"m"}, {{"MAKO_ZZ_COUNT", "11"}, {"MAKO_ZZ_TEXT", "from env"}});
        check(r.t.count == 11 && r.t.text == "from env", "variables give values");
        check(run({"m", "--zz-count", "2"}, {{"MAKO_ZZ_COUNT", "11"}}).t.count == 2, "the command line wins");
        check(run({"m"}, {{"MAKO_ZZ_COUNT", ""}}).t.count == 0, "an empty variable is ignored");
        check(run({"m"}, {{"MAKO_ZZ_COUNT", "x"}}).error == "--zz-count takes a non-negative integer (got x from MAKO_ZZ_COUNT)",
              "a bad value names its variable");
        check(!run({"m"}, {{"MAKO_ZZ_SWITCH", "0"}}).t.sw && !run({"m"}, {{"MAKO_ZZ_SWITCH", "off"}}).t.sw &&
                  run({"m"}, {{"MAKO_ZZ_SWITCH", "yes"}}).t.sw,
              "a flag's variable: 0 / off clear, yes sets");
        check(run({"m", "--zz-switch"}, {{"MAKO_ZZ_SWITCH", "0"}}).t.sw, "the command line's flag wins");
        check(run({"m"}, {{"MAKO_ZZ_SWITCH", "maybe"}}).error == "MAKO_ZZ_SWITCH=maybe takes 1 or 0", "a bad flag value");
        check(run({"m"}, {{"MAKO_N", "4"}}).t.short_n == 0, "short options have no variable");
        check(run({"m", "--zz-key", "a"}, {{"MAKO_ZZ_KEY", "e"}}).t.keys == std::vector<std::string>{"a"},
              "an option that adds up: the command line's replace the variable's");
        check(run({"m", "--zz-key-file", "f"}, {{"MAKO_ZZ_KEY", "e"}}).t.keys == std::vector<std::string>{"file:f"},
              "and so do those of its group");
        check(run({"m"}, {{"MAKO_ZZ_KEY", "e"}, {"MAKO_ZZ_KEY_FILE", "f"}}).t.keys ==
                  std::vector<std::string>{"e", "file:f"},
              "without the command line, both variables apply");
        check(run({"m"}, {{"MAKO_ZZ_KEY", "s3cret"}, {"MAKO_ZZ_COUNT", "x"}}).error.find("s3cret") == std::string::npos,
              "a secret from the environment stays out of errors");
    }

    std::println("presets");
    {
        const auto file = dir / "cli_test_presets.json";
        std::ofstream(file) << R"({
          "p": { "model": "/preset.gguf", "--zz-count": 5, "--zz-text": "preset", "--no-zz-switch": true },
          "q": { "--zz-count": 6 }
        })";
        const std::string f = file.string();
        auto r = run({"--presets", f, "--preset", "p"});
        check(r.positionals == std::vector<std::string>{"/preset.gguf"} && r.t.count == 5 && r.t.text == "preset" &&
                  !r.t.sw,
              "a preset gives options and the model");
        r = run({"m", "--presets", f, "--preset", "p", "--zz-count", "8"});
        check(r.positionals == std::vector<std::string>{"m"} && r.t.count == 8, "options after a preset override it");
        check(run({"m", "--presets", f, "--preset", "p"}, {{"MAKO_ZZ_COUNT", "11"}, {"MAKO_ZZ_RATIO", "0.5"}}).t.count == 5,
              "a preset overrides the environment");
        check(run({"m", "--presets", f, "--preset", "p"}, {{"MAKO_ZZ_RATIO", "0.5"}}).t.ratio == 0.5,
              "variables for what the preset leaves apply");
        r = run({"m"}, {{"MAKO_PRESETS", f}, {"MAKO_PRESET", "p"}});
        check(r.error.empty() && r.positionals == std::vector<std::string>{"m"} && r.t.count == 5,
              "MAKO_PRESETS and MAKO_PRESET");
        r = run({}, {{"MAKO_PRESETS", f}, {"MAKO_PRESET", "p"}});
        check(r.positionals == std::vector<std::string>{"/preset.gguf"}, "MAKO_PRESET's model");
        check(run({"m", "--preset", "q"}, {{"MAKO_PRESETS", f}, {"MAKO_PRESET", "p"}}).t.count == 6 &&
                  run({"m", "--preset", "q"}, {{"MAKO_PRESETS", f}, {"MAKO_PRESET", "p"}}).t.text == "unset",
              "--preset replaces MAKO_PRESET");
        check(run({"m", "--zz-count", "9"}, {{"MAKO_PRESETS", f}, {"MAKO_PRESET", "p"}}).t.count == 9,
              "the command line overrides MAKO_PRESET");
        check(run({"m", "--preset", "nope"}, {{"MAKO_PRESETS", f}}).error.starts_with("no preset nope"),
              "an unknown preset");
    }

    if (failures) {
        std::println("{} failures", failures);
        return 1;
    }
    std::println("cli: all cases pass");
    return 0;
}
