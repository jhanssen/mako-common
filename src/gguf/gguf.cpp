#include "gguf/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <format>

namespace mako::gguf {

TypeTraits traits(Type t)
{
    switch (t) {
    case Type::F32: return {"F32", 1, 4};
    case Type::F16: return {"F16", 1, 2};
    case Type::BF16: return {"BF16", 1, 2};
    case Type::F64: return {"F64", 1, 8};
    case Type::I8: return {"I8", 1, 1};
    case Type::I16: return {"I16", 1, 2};
    case Type::I32: return {"I32", 1, 4};
    case Type::I64: return {"I64", 1, 8};
    case Type::Q4_0: return {"Q4_0", 32, 18};
    case Type::Q4_1: return {"Q4_1", 32, 20};
    case Type::Q5_0: return {"Q5_0", 32, 22};
    case Type::Q5_1: return {"Q5_1", 32, 24};
    case Type::Q8_0: return {"Q8_0", 32, 34};
    case Type::Q8_1: return {"Q8_1", 32, 36};
    case Type::Q2_K: return {"Q2_K", 256, 84};
    case Type::Q3_K: return {"Q3_K", 256, 110};
    case Type::Q4_K: return {"Q4_K", 256, 144};
    case Type::Q5_K: return {"Q5_K", 256, 176};
    case Type::Q6_K: return {"Q6_K", 256, 210};
    case Type::Q8_K: return {"Q8_K", 256, 292};
    case Type::IQ2_XXS: return {"IQ2_XXS", 256, 66};
    case Type::IQ2_XS: return {"IQ2_XS", 256, 74};
    case Type::IQ2_S: return {"IQ2_S", 256, 82};
    case Type::IQ3_XXS: return {"IQ3_XXS", 256, 98};
    case Type::IQ3_S: return {"IQ3_S", 256, 110};
    case Type::IQ1_S: return {"IQ1_S", 256, 50};
    case Type::IQ1_M: return {"IQ1_M", 256, 56};
    case Type::IQ4_NL: return {"IQ4_NL", 32, 18};
    case Type::IQ4_XS: return {"IQ4_XS", 256, 136};
    case Type::TQ1_0: return {"TQ1_0", 256, 54};
    case Type::TQ2_0: return {"TQ2_0", 256, 66};
    case Type::MXFP4: return {"MXFP4", 32, 17};
    case Type::MXFP8: return {"MXFP8", 32, 33};
    }
    return {"?", 0, 0};
}

uint64_t Tensor::elements() const
{
    uint64_t n = 1;
    for (uint64_t d : dims)
        n *= d;
    return n;
}

namespace {

enum class VType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

struct ParseError {
    std::string msg;
};

class Reader {
public:
    Reader(const std::byte* p, uint64_t n) : p_(p), n_(n) {}

    uint64_t pos() const { return pos_; }

    template <typename T>
    T get()
    {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string str()
    {
        const uint64_t len = get<uint64_t>();
        need(len);
        std::string s(reinterpret_cast<const char*>(p_ + pos_), len);
        pos_ += len;
        return s;
    }

    Value value(VType t, int depth = 0)
    {
        switch (t) {
        case VType::U8: return uint64_t{get<uint8_t>()};
        case VType::I8: return int64_t{get<int8_t>()};
        case VType::U16: return uint64_t{get<uint16_t>()};
        case VType::I16: return int64_t{get<int16_t>()};
        case VType::U32: return uint64_t{get<uint32_t>()};
        case VType::I32: return int64_t{get<int32_t>()};
        case VType::U64: return get<uint64_t>();
        case VType::I64: return get<int64_t>();
        case VType::F32: return double{get<float>()};
        case VType::F64: return get<double>();
        case VType::BOOL: return get<uint8_t>() != 0;
        case VType::STRING: return str();
        case VType::ARRAY: {
            if (depth > 4)
                throw ParseError{"nested arrays too deep"};
            const auto et = static_cast<VType>(get<uint32_t>());
            const uint64_t count = get<uint64_t>();
            if (count > n_ - pos_)
                throw ParseError{std::format("array of {} items at {} overruns the file", count, pos_)};
            Array a;
            a.items.reserve(count);
            for (uint64_t i = 0; i < count; ++i)
                a.items.push_back(value(et, depth + 1));
            return a;
        }
        }
        throw ParseError{std::format("unknown value type {} at {}", static_cast<uint32_t>(t), pos_)};
    }

private:
    void need(uint64_t k) const
    {
        if (k > n_ - pos_)
            throw ParseError{std::format("truncated header at offset {}", pos_)};
    }

    const std::byte* p_;
    uint64_t n_;
    uint64_t pos_ = 0;
};

} // namespace

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
    f.size_ = static_cast<uint64_t>(st.st_size);
    void* m = ::mmap(nullptr, f.size_, PROT_READ, MAP_SHARED, f.fd_, 0);
    if (m == MAP_FAILED)
        return std::unexpected(std::format("mmap {}: {}", path, std::strerror(errno)));
    f.map_ = static_cast<const std::byte*>(m);

    try {
        Reader r(f.map_, f.size_);
        if (r.get<uint32_t>() != 0x46554747u)
            return std::unexpected(path + ": not a GGUF file");
        f.version_ = r.get<uint32_t>();
        if (f.version_ < 2 || f.version_ > 3)
            return std::unexpected(std::format("{}: unsupported GGUF version {}", path, f.version_));
        const uint64_t n_tensors = r.get<uint64_t>();
        const uint64_t n_kv = r.get<uint64_t>();

        f.kv_.reserve(n_kv);
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = r.str();
            Value v = r.value(static_cast<VType>(r.get<uint32_t>()));
            f.kv_.emplace_back(std::move(key), std::move(v));
        }

        f.tensors_.reserve(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            Tensor t;
            t.name = r.str();
            const uint32_t nd = r.get<uint32_t>();
            if (nd == 0 || nd > 8)
                return std::unexpected(std::format("{}: tensor {} has {} dims", path, t.name, nd));
            for (uint32_t d = 0; d < nd; ++d)
                t.dims.push_back(r.get<uint64_t>());
            t.type = static_cast<Type>(r.get<uint32_t>());
            t.offset = r.get<uint64_t>();
            const TypeTraits tr = traits(t.type);
            if (tr.block_elems == 0)
                return std::unexpected(std::format("{}: tensor {} has unknown type {}", path, t.name,
                                                   static_cast<uint32_t>(t.type)));
            if (t.dims[0] % tr.block_elems != 0)
                return std::unexpected(std::format("{}: tensor {} row of {} is not whole {} blocks", path, t.name,
                                                   t.dims[0], tr.name));
            t.bytes = t.elements() / tr.block_elems * tr.block_bytes;
            f.tensors_.push_back(std::move(t));
        }

        uint64_t align = 32;
        if (auto a = f.get_int("general.alignment"))
            align = static_cast<uint64_t>(*a);
        if (align == 0 || (align & (align - 1)) != 0)
            return std::unexpected(std::format("{}: bad alignment {}", path, align));
        f.data_offset_ = (r.pos() + align - 1) / align * align;
        for (auto& t : f.tensors_) {
            if (t.offset % align != 0)
                return std::unexpected(std::format("{}: tensor {} offset {} is not aligned", path, t.name, t.offset));
            t.offset += f.data_offset_;
        }
    } catch (const ParseError& e) {
        return std::unexpected(path + ": " + e.msg);
    }

    for (size_t i = 0; i < f.tensors_.size(); ++i)
        f.by_name_.emplace(f.tensors_[i].name, i);
    return f;
}

File::File(File&& o) noexcept { *this = std::move(o); }

File& File::operator=(File&& o) noexcept
{
    if (this != &o) {
        unmap();
        path_ = std::move(o.path_);
        fd_ = std::exchange(o.fd_, -1);
        map_ = std::exchange(o.map_, nullptr);
        size_ = o.size_;
        version_ = o.version_;
        data_offset_ = o.data_offset_;
        kv_ = std::move(o.kv_);
        tensors_ = std::move(o.tensors_);
        by_name_.clear();
        o.by_name_.clear();
        for (size_t i = 0; i < tensors_.size(); ++i)
            by_name_.emplace(tensors_[i].name, i);
    }
    return *this;
}

File::~File() { unmap(); }

void File::unmap()
{
    if (map_)
        ::munmap(const_cast<std::byte*>(map_), size_);
    if (fd_ >= 0)
        ::close(fd_);
    map_ = nullptr;
    fd_ = -1;
}

const Value* File::find(std::string_view key) const
{
    for (const auto& [k, v] : kv_)
        if (k == key)
            return &v;
    return nullptr;
}

std::optional<int64_t> File::get_int(std::string_view key) const
{
    const Value* v = find(key);
    if (!v)
        return std::nullopt;
    if (auto* u = std::get_if<uint64_t>(v))
        return static_cast<int64_t>(*u);
    if (auto* i = std::get_if<int64_t>(v))
        return *i;
    return std::nullopt;
}

std::optional<double> File::get_float(std::string_view key) const
{
    const Value* v = find(key);
    if (!v)
        return std::nullopt;
    if (auto* d = std::get_if<double>(v))
        return *d;
    if (auto i = get_int(key))
        return static_cast<double>(*i);
    return std::nullopt;
}

std::optional<std::string_view> File::get_string(std::string_view key) const
{
    const Value* v = find(key);
    if (auto* s = v ? std::get_if<std::string>(v) : nullptr)
        return *s;
    return std::nullopt;
}

const Tensor* File::tensor(std::string_view name) const
{
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

std::span<const std::byte> File::data(const Tensor& t) const
{
    if (t.offset + t.bytes > size_)
        return {};
    return {map_ + t.offset, t.bytes};
}

std::string to_string(const Value& v, size_t max_items)
{
    struct Visitor {
        size_t max_items;
        std::string operator()(uint64_t x) const { return std::to_string(x); }
        std::string operator()(int64_t x) const { return std::to_string(x); }
        std::string operator()(double x) const { return std::format("{}", x); }
        std::string operator()(bool x) const { return x ? "true" : "false"; }
        std::string operator()(const std::string& s) const
        {
            return s.size() > 80 ? std::format("\"{}...\" ({} chars)", s.substr(0, 80), s.size()) : "\"" + s + "\"";
        }
        std::string operator()(const Array& a) const
        {
            std::string out = "[";
            for (size_t i = 0; i < a.items.size() && i < max_items; ++i) {
                if (i)
                    out += ", ";
                out += std::visit(*this, a.items[i]);
            }
            if (a.items.size() > max_items)
                out += std::format(", ... ({} items)", a.items.size());
            return out + "]";
        }
    };
    return std::visit(Visitor{max_items}, v);
}

} // namespace mako::gguf
