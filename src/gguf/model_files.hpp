#pragma once

#include "gguf/gguf.hpp"

#include <deque>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace mako::gguf {

// One GGUF model, possibly split into "-0000N-of-0000M.gguf" shards. Metadata comes from the first shard.
class ModelFiles {
public:
    struct Ref {
        const File* file = nullptr;
        const Tensor* tensor = nullptr;
        explicit operator bool() const { return tensor != nullptr; }
    };

    static std::expected<ModelFiles, std::string> open(const std::string& first_shard);
    // Adds one more GGUF (tensors kept elsewhere, such as a separate set of tables); find() searches it too.
    std::expected<void, std::string> add(const std::string& path);

    const File& meta() const { return files_.front(); }
    const std::deque<File>& files() const { return files_; }
    Ref find(std::string_view name) const;
    uint64_t tensor_count() const;

private:
    std::deque<File> files_; // (a deque: add() leaves references to the others valid)
};

} // namespace mako::gguf
