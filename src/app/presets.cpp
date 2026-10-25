#include "app/presets.hpp"

#include <glaze/glaze.hpp>

#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <variant>

namespace mako::app {

namespace {

using Value = std::variant<std::nullptr_t, bool, double, std::string, std::vector<std::string>>;
using Presets = std::map<std::string, std::map<std::string, Value>>;

std::string home_path(const std::string& s)
{
    if (s.starts_with("~/"))
        if (const char* home = std::getenv("HOME"))
            return std::string(home) + s.substr(1);
    return s;
}

std::string number(double v)
{
    if (v == double(int64_t(v)))
        return std::to_string(int64_t(v));
    return std::format("{}", v);
}

Presets load(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error(std::format("cannot open presets file {}", path.string()));
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Presets p;
    if (auto ec = glz::read_json(p, text))
        throw std::runtime_error(std::format("{}: {}", path.string(), glz::format_error(ec, text)));
    return p;
}

} // namespace

std::filesystem::path presets_path()
{
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : ".") / "Library/Application Support/mako/presets.json";
}

Args::Args(int argc, char** argv)
{
    std::filesystem::path file = presets_path();
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == "--presets")
            file = home_path(argv[i + 1]);
    std::vector<std::string> out{argv[0]}, models;
    bool positional = argc > 1 && !std::string_view(argv[1]).starts_with("--");
    Presets presets;
    bool loaded = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--presets" && i + 1 < argc) {
            ++i;
            continue;
        }
        if (a != "--preset" || i + 1 >= argc) {
            out.push_back(a);
            continue;
        }
        if (!loaded) {
            presets = load(file);
            loaded = true;
        }
        const std::string name = argv[++i];
        const auto it = presets.find(name);
        if (it == presets.end())
            throw std::runtime_error(std::format("no preset {} in {}", name, file.string()));
        for (const auto& [key, value] : it->second) {
            if (key == "model") {
                if (const auto* s = std::get_if<std::string>(&value))
                    models = {home_path(*s)};
                else if (const auto* v = std::get_if<std::vector<std::string>>(&value))
                    for (models.clear(); const auto& m : *v)
                        models.push_back(home_path(m));
                continue;
            }
            if (!key.starts_with("--"))
                continue; // description and other notes
            if (const auto* b = std::get_if<bool>(&value)) {
                if (*b)
                    out.push_back(key);
            } else if (const auto* d = std::get_if<double>(&value)) {
                out.push_back(key);
                out.push_back(number(*d));
            } else if (const auto* s = std::get_if<std::string>(&value)) {
                out.push_back(key);
                out.push_back(home_path(*s));
            }
        }
    }
    if (!positional && !models.empty())
        out.insert(out.begin() + 1, models.begin(), models.end());
    args_ = std::move(out);
    for (auto& s : args_)
        ptrs_.push_back(s.data());
    ptrs_.push_back(nullptr);
}

} // namespace mako::app
