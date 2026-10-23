#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mako::gguf {

// ggml tensor types (values are the file format's).
enum class Type : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
};

struct TypeTraits {
    std::string_view name;
    uint32_t block_elems = 0; // 0: unknown type
    uint32_t block_bytes = 0;
};

TypeTraits traits(Type t);

struct Array;
using Value = std::variant<uint64_t, int64_t, double, bool, std::string, Array>;

struct Array {
    std::vector<Value> items;
};

struct Tensor {
    std::string name;
    Type type;
    std::vector<uint64_t> dims; // ggml order: dims[0] is the contiguous one
    uint64_t offset = 0;        // absolute file offset
    uint64_t bytes = 0;

    uint64_t elements() const;
};

class File {
public:
    static std::expected<File, std::string> open(const std::string& path);

    File(File&&) noexcept;
    File& operator=(File&&) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File();

    const std::string& path() const { return path_; }
    uint64_t size() const { return size_; }
    uint32_t version() const { return version_; }
    uint64_t data_offset() const { return data_offset_; }

    const std::vector<std::pair<std::string, Value>>& metadata() const { return kv_; }
    const Value* find(std::string_view key) const;
    std::optional<int64_t> get_int(std::string_view key) const;
    std::optional<double> get_float(std::string_view key) const;
    std::optional<std::string_view> get_string(std::string_view key) const;

    const std::vector<Tensor>& tensors() const { return tensors_; }
    const Tensor* tensor(std::string_view name) const;

    // Bytes of a tensor in the mapped file; empty when the file is shorter than the tensor's end.
    std::span<const std::byte> data(const Tensor& t) const;

private:
    File() = default;
    void unmap();

    std::string path_;
    int fd_ = -1;
    const std::byte* map_ = nullptr;
    uint64_t size_ = 0;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    std::vector<std::pair<std::string, Value>> kv_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string_view, size_t> by_name_;
};

std::string to_string(const Value& v, size_t max_items = 8);

} // namespace mako::gguf
