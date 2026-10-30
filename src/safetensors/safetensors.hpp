#pragma once

#include "gguf/gguf.hpp"

#include <deque>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mako::safetensors {

// One .safetensors file: its tensors described as GGUF tensors are (dims[0] the contiguous one, absolute offsets),
// read through fd().
class File {
public:
    static std::expected<File, std::string> open(const std::string& path);

    File(File&&) noexcept;
    File& operator=(File&&) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File();

    const std::string& path() const { return path_; }
    int fd() const { return fd_; }
    const std::vector<gguf::Tensor>& tensors() const { return tensors_; }
    const gguf::Tensor* tensor(std::string_view name) const;
    // The header's __metadata__ (strings).
    const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

private:
    File() = default;

    std::string path_;
    int fd_ = -1;
    std::vector<gguf::Tensor> tensors_;
    std::unordered_map<std::string, size_t> by_name_;
    std::unordered_map<std::string, std::string> metadata_;
};

// A model's safetensors: one file, or a directory holding one file or an index (`*.safetensors.index.json`) of
// several.
class Files {
public:
    struct Ref {
        const File* file = nullptr;
        const gguf::Tensor* tensor = nullptr;
        explicit operator bool() const { return tensor != nullptr; }
    };

    static std::expected<Files, std::string> open(const std::string& path);
    Ref find(std::string_view name) const;
    const std::deque<File>& files() const { return files_; }

private:
    std::deque<File> files_;
};

} // namespace mako::safetensors
