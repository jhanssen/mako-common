// Preset expansion (app::Args): flags in place of --preset, overrides after it, model paths as positionals.
//   presets_test <scratch dir>
#include "app/presets.hpp"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const std::filesystem::path dir = argc > 1 ? argv[1] : std::filesystem::temp_directory_path();
    const auto file = dir / "presets_test.json";
    std::ofstream(file) << R"({
      "a": { "description": "x", "model": "/m.gguf", "--ctx": 4096, "--miss-penalty": 0.1, "--no-spec": true,
             "--learn": false, "--engram": "/e.gguf" },
      "b": { "model": ["/one", "/two"], "--port": 9000 }
    })";
    int failures = 0;
    auto expand = [&](std::vector<std::string> in) {
        std::vector<char*> ptrs;
        for (auto& s : in)
            ptrs.push_back(s.data());
        ptrs.push_back(nullptr);
        mako::app::Args args(int(in.size()), ptrs.data());
        std::vector<std::string> out(args.argv(), args.argv() + args.argc());
        return out;
    };
    auto check = [&](const std::vector<std::string>& got, const std::vector<std::string>& want, const std::string& what) {
        const bool ok = got == want;
        std::string g;
        for (const auto& s : got)
            g += s + " ";
        std::println("  {}  {}: {}", ok ? "ok  " : "FAIL", what, g);
        failures += !ok;
    };
    const std::string f = file.string();
    // Flags sorted by name within a preset; false and notes left out.
    check(expand({"tool", "--presets", f, "--preset", "a"}),
          {"tool", "/m.gguf", "--ctx", "4096", "--engram", "/e.gguf", "--miss-penalty", "0.1", "--no-spec"},
          "model inserted, flags expanded");
    check(expand({"tool", "/given.gguf", "--presets", f, "--preset", "a", "--ctx", "100"}),
          {"tool", "/given.gguf", "--ctx", "4096", "--engram", "/e.gguf", "--miss-penalty", "0.1", "--no-spec", "--ctx",
           "100"},
          "positional kept, later option after the preset");
    check(expand({"tool", "--presets", f, "--preset", "b"}), {"tool", "/one", "/two", "--port", "9000"},
          "several model paths");
    check(expand({"tool", "x", "--port", "1"}), {"tool", "x", "--port", "1"}, "no preset: unchanged");
    bool threw = false;
    try {
        expand({"tool", "--presets", f, "--preset", "missing"});
    } catch (const std::exception&) {
        threw = true;
    }
    std::println("  {}  unknown preset throws", threw ? "ok  " : "FAIL");
    failures += !threw;
    std::filesystem::remove(file);
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
