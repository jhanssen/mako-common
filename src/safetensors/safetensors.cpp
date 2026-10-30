#include "safetensors/safetensors.hpp"

#include "gguf/read_pool.hpp"

#include <glaze/glaze.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>

namespace mako::safetensors {

namespace {

namespace fs = std::filesystem;

std::optional<gguf::Type> type_of(std::string_view dtype)
{
    if (dtype == "BF16")
        return gguf::Type::BF16;
    if (dtype == "F16")
        return gguf::Type::F16;
    if (dtype == "F32")
        return gguf::Type::F32;
    return std::nullopt;
}

} // namespace

File::File(File&& o) noexcept
    : path_(std::move(o.path_)), fd_(std::exchange(o.fd_, -1)), tensors_(std::move(o.tensors_)),
      by_name_(std::move(o.by_name_)), metadata_(std::move(o.metadata_))
{
}

File& File::operator=(File&& o) noexcept
{
    if (this != &o) {
        if (fd_ >= 0)
            ::close(fd_);
        path_ = std::move(o.path_);
        fd_ = std::exchange(o.fd_, -1);
        tensors_ = std::move(o.tensors_);
        by_name_ = std::move(o.by_name_);
        metadata_ = std::move(o.metadata_);
    }
    return *this;
}

File::~File()
{
    if (fd_ >= 0)
        ::close(fd_);
}

// An 8-byte little-endian header length, the JSON header ({name: {dtype, shape, data_offsets}, "__metadata__": ...}),
// then the data, offsets relative to its start; shapes are row-major (the last dimension contiguous).
std::expected<File, std::string> File::open(const std::string& path)
{
    File f;
    f.path_ = path;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0)
        return std::unexpected(std::format("open {}: {}", path, std::strerror(errno)));
    struct stat st{};
    if (::fstat(f.fd_, &st) != 0)
        return std::unexpected(std::format("stat {}: {}", path, std::strerror(errno)));
    const uint64_t size = uint64_t(st.st_size);
    uint64_t header = 0;
    std::string text;
    try {
        if (size >= 8)
            gguf::read_exact(f.fd_, &header, 8, 0);
        if (size < 8 || header > size - 8 || header > (100ull << 20))
            return std::unexpected(path + ": not a safetensors file");
        text.resize(header);
        gguf::read_exact(f.fd_, text.data(), header, 8);
    } catch (const std::exception& e) {
        return std::unexpected(std::format("{}: {}", path, e.what()));
    }
    glz::generic j;
    if (glz::read_json(j, text) || !j.is_object())
        return std::unexpected(path + ": bad header");
    const uint64_t data = 8 + header;
    for (const auto& [name, t] : j.get_object()) {
        if (name == "__metadata__") {
            if (t.is_object())
                for (const auto& [k, v] : t.get_object())
                    if (v.is_string())
                        f.metadata_.emplace(k, v.get_string());
            continue;
        }
        if (!t.is_object() || !t.contains("dtype") || !t.contains("shape") || !t.contains("data_offsets"))
            return std::unexpected(std::format("{}: bad entry {}", path, name));
        const auto type = type_of(t["dtype"].get_string());
        if (!type)
            return std::unexpected(std::format("{}: {} has unsupported dtype {}", path, name, t["dtype"].get_string()));
        gguf::Tensor g;
        g.name = name;
        g.type = *type;
        const auto& shape = t["shape"].get_array();
        for (auto it = shape.rbegin(); it != shape.rend(); ++it)
            g.dims.push_back(uint64_t(it->get_number()));
        if (g.dims.empty())
            g.dims.push_back(1);
        const auto& off = t["data_offsets"].get_array();
        if (off.size() != 2)
            return std::unexpected(std::format("{}: bad offsets of {}", path, name));
        const uint64_t begin = uint64_t(off[0].get_number()), end = uint64_t(off[1].get_number());
        g.offset = data + begin;
        g.bytes = end - begin;
        if (end < begin || g.offset + g.bytes > size || g.bytes != g.elements() * gguf::traits(g.type).block_bytes)
            return std::unexpected(std::format("{}: {} does not fit its shape or the file", path, name));
        f.by_name_.emplace(name, f.tensors_.size());
        f.tensors_.push_back(std::move(g));
    }
    return f;
}

const gguf::Tensor* File::tensor(std::string_view name) const
{
    const auto it = by_name_.find(std::string(name));
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

std::expected<Files, std::string> Files::open(const std::string& path)
{
    std::vector<std::string> paths;
    if (fs::is_directory(path)) {
        std::optional<fs::path> index;
        std::vector<fs::path> single;
        for (const auto& e : fs::directory_iterator(path)) {
            const std::string n = e.path().filename().string();
            if (n.ends_with(".safetensors.index.json"))
                index = e.path();
            else if (n.ends_with(".safetensors"))
                single.push_back(e.path());
        }
        if (index) {
            std::ifstream in(*index, std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            glz::generic j;
            if (glz::read_json(j, text) || !j.is_object() || !j.contains("weight_map") || !j["weight_map"].is_object())
                return std::unexpected(index->string() + ": bad index");
            std::set<std::string> shards;
            for (const auto& [name, file] : j["weight_map"].get_object())
                shards.insert(file.get_string());
            for (const auto& s : shards)
                paths.push_back((fs::path(path) / s).string());
        } else if (single.size() == 1) {
            paths.push_back(single.front().string());
        } else {
            return std::unexpected(std::format("{}: {} safetensors files and no index", path, single.size()));
        }
    } else {
        paths.push_back(path);
    }
    Files m;
    for (const auto& p : paths) {
        auto f = File::open(p);
        if (!f)
            return std::unexpected(f.error());
        m.files_.push_back(std::move(*f));
    }
    return m;
}

Files::Ref Files::find(std::string_view name) const
{
    for (const File& f : files_)
        if (const gguf::Tensor* t = f.tensor(name))
            return {&f, t};
    return {};
}

} // namespace mako::safetensors
