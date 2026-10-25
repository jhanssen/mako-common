#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace mako::app {

// Named sets of command-line options in a JSON file (default presets_path()), applied with --preset NAME:
//   { "ds-fast": { "description": "...", "model": "~/models/x.gguf", "--miss-penalty": 0.1, "--no-spec": true } }
// Each "--flag" entry becomes "--flag value" (true: the flag alone; false or null: left out), inserted where
// --preset stood, so options after it override it and several presets apply in order. "model" (a path, or an array
// of paths for tools taking several) fills the leading positional arguments when the command line gives none.
// A leading "~/" in a value is the home directory. --presets FILE names another file.
class Args {
public:
    // Throws on an unknown preset or an unreadable file.
    Args(int argc, char** argv);

    int argc() const { return int(ptrs_.size()) - 1; }
    char** argv() { return ptrs_.data(); }

private:
    std::vector<std::string> args_;
    std::vector<char*> ptrs_;
};

// ~/Library/Application Support/mako/presets.json
std::filesystem::path presets_path();

} // namespace mako::app
