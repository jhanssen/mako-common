#pragma once

#include "gguf/gguf.hpp"

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

    const File& meta() const { return files_.front(); }
    const std::vector<File>& files() const { return files_; }
    Ref find(std::string_view name) const;
    uint64_t tensor_count() const;

private:
    std::vector<File> files_;
};

} // namespace mako::gguf
