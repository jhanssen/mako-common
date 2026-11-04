#include "models/runtime.hpp"

#include "gguf/read_pool.hpp"

#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::models {

namespace {

struct HeadsArgs {
    uint32_t rows, ld, col0, dim, n_pad, row0, rope;
};
struct HeadsNormArgs {
    uint32_t ld, col0, dim, n_pad, row0, rope;
    float eps;
};
struct HeadsPartialArgs {
    uint32_t ld, col0, stride, dim, rot, n_pad;
    float eps;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // Round to nearest even; NaN stays NaN.
    if ((u & 0x7f800000) == 0x7f800000 && (u & 0x007fffff))
        return uint16_t((u >> 16) | 0x40);
    const uint32_t lsb = (u >> 16) & 1;
    return uint16_t((u + 0x7fff + lsb) >> 16);
}

} // namespace

std::vector<float> read_floats(const safetensors::File& f, const gguf::Tensor& t)
{
    const uint64_t n = t.elements();
    std::vector<float> out(n);
    std::vector<std::byte> raw(t.bytes);
    gguf::read_exact(f.fd(), raw.data(), t.bytes, t.offset);
    for (uint64_t i = 0; i < n; ++i) {
        if (t.type == gguf::Type::F32) {
            std::memcpy(&out[i], raw.data() + i * 4, 4);
        } else if (t.type == gguf::Type::F16) {
            __fp16 h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            out[i] = float(h);
        } else if (t.type == gguf::Type::BF16) {
            uint16_t b;
            std::memcpy(&b, raw.data() + i * 2, 2);
            const uint32_t u = uint32_t(b) << 16;
            std::memcpy(&out[i], &u, 4);
        } else {
            throw std::runtime_error(std::format("{}: not an F32, F16 or BF16 tensor", t.name));
        }
    }
    return out;
}

std::vector<float> TransformerRuntime::floats(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    return read_floats(*r.file, *r.tensor);
}

TransformerRuntime::Tensor TransformerRuntime::upload_rows(const safetensors::Files& files,
                                                           std::span<const std::string> names)
{
    uint64_t rows = 0, k = 0;
    bool f16 = true;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (!r ||
            (r.tensor->type != gguf::Type::BF16 && r.tensor->type != gguf::Type::F32 &&
             r.tensor->type != gguf::Type::F16) ||
            r.tensor->dims.size() != 2 || (k && r.tensor->dims[0] != k))
            throw std::runtime_error(n + ": missing, or not a BF16, F16 or F32 matrix of the others' width");
        k = r.tensor->dims[0];
        rows += r.tensor->dims[1];
        f16 = f16 && r.tensor->type == gguf::Type::F16;
    }
    Tensor t = reserve(rows * k * 2, f16 ? gguf::Type::F16 : gguf::Type::BF16, {k, rows});
    char* dst = static_cast<char*>(t.buf->contents()) + t.off;
    for (const auto& n : names) {
        const auto r = files.find(n);
        if (r.tensor->type == gguf::Type::BF16 || f16) {
            gguf::read_exact(r.file->fd(), dst, r.tensor->bytes, r.tensor->offset);
            dst += r.tensor->bytes;
        } else {
            const auto f = read_floats(*r.file, *r.tensor);
            auto* o = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < f.size(); ++i)
                o[i] = to_bf16(f[i]);
            dst += f.size() * 2;
        }
    }
    return t;
}

TransformerRuntime::Tensor TransformerRuntime::upload(const safetensors::Files& files, const std::string& name,
                                                      bool matrix)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    if (matrix && r.tensor->type == gguf::Type::F32 && r.tensor->dims.size() == 2) {
        const std::string n[1] = {name};
        return upload_rows(files, n);
    }
    return upload_raw(r.file->fd(), r.tensor->offset, r.tensor->bytes, r.tensor->type, r.tensor->dims, name, matrix);
}

TransformerRuntime::Tensor TransformerRuntime::upload_f32(const safetensors::Files& files, const std::string& name)
{
    const auto f = floats(files, name);
    Tensor t = reserve(f.size() * 4, gguf::Type::F32, {f.size()});
    std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, f.data(), f.size() * 4);
    return t;
}

TransformerRuntime::Tensor TransformerRuntime::upload_f16(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r || r.tensor->dims.size() != 2)
        throw std::runtime_error(name + ": missing, or not a matrix");
    const auto f = read_floats(*r.file, *r.tensor);
    Tensor t = reserve(f.size() * 2, gguf::Type::F16, r.tensor->dims);
    auto* o = reinterpret_cast<__fp16*>(static_cast<char*>(t.buf->contents()) + t.off);
    for (size_t i = 0; i < f.size(); ++i)
        o[i] = __fp16(f[i]);
    return t;
}

TransformerRuntime::Tensor TransformerRuntime::upload_plus_one(const safetensors::Files& files, const std::string& name)
{
    const auto r = files.find(name);
    if (!r)
        throw std::runtime_error("missing tensor " + name);
    const auto f = read_floats(*r.file, *r.tensor);
    Tensor t = reserve(f.size() * 4, gguf::Type::F32, {f.size()});
    auto* o = reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off);
    for (size_t i = 0; i < f.size(); ++i)
        o[i] = f[i] + 1.0f;
    return t;
}

void TransformerRuntime::dispatch_at(MTL::ComputeCommandEncoder*& enc, const std::string& fn,
                                     std::initializer_list<Arg> args, const void* bytes, size_t bytes_size,
                                     MTL::Size grid, MTL::Size tg, bool threads)
{
    enc->setComputePipelineState(pso(fn));
    NS::UInteger i = 0;
    for (const Arg& a : args)
        enc->setBuffer(a.buf, a.off, i++);
    if (bytes)
        enc->setBytes(bytes, bytes_size, i);
    if (threads)
        enc->dispatchThreads(grid, tg);
    else
        enc->dispatchThreadgroups(grid, tg);
    after(enc, fn);
}

void TransformerRuntime::add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y,
                             uint32_t n)
{
    dispatch(enc, "add", {x, z, y}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void TransformerRuntime::unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n)
{
    dispatch(enc, fn, {x}, &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

void TransformerRuntime::bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n,
                              uint32_t rows, uint64_t y_off)
{
    enc->setComputePipelineState(pso("bias_rows"));
    enc->setBuffer(y, y_off, 0);
    enc->setBuffer(b.buf, b.off, 1);
    enc->setBytes(&n, sizeof(n), 2);
    enc->dispatchThreads(MTL::Size(n, rows, 1), MTL::Size(64, 1, 1));
    after(enc, "bias_rows");
}

void TransformerRuntime::heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0,
                               uint32_t n_heads, uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope,
                               MTL::Buffer* y)
{
    const HeadsArgs a{T, ld, col0, kHeadDim, n_pad, 0, rope};
    dispatch(enc, "heads_half", {x, cs ? cs : x, y}, &a, sizeof(a), MTL::Size(kHeadDim / 2, T, n_heads),
             MTL::Size(64, 1, 1));
    if (n_pad > T) {
        const uint32_t c[4] = {kHeadDim, n_pad, T, 0};
        dispatch(enc, "heads_clear", {y}, c, sizeof(c), MTL::Size(kHeadDim, n_pad - T, n_heads), MTL::Size(128, 1, 1));
    }
}

void TransformerRuntime::heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0,
                                    uint32_t n_heads, uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off,
                                    uint32_t rope, const Tensor& w, float eps, uint32_t row0, MTL::Buffer* y)
{
    const HeadsNormArgs a{ld, col0, kHeadDim, n_pad, row0, rope, eps};
    enc->setComputePipelineState(pso("heads_norm_half"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(cs ? cs : x, cs ? cs_off : 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBuffer(w.buf, w.off, 3);
    enc->setBytes(&a, sizeof(a), 4);
    enc->dispatchThreadgroups(MTL::Size(T, n_heads, 1), MTL::Size(kHeadDim / 2, 1, 1));
    after(enc, "heads_norm_half");
}

void TransformerRuntime::heads_partial(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0,
                                       uint32_t stride, uint32_t dim, uint32_t rot, uint32_t n_heads, uint32_t T,
                                       uint32_t n_pad, MTL::Buffer* cs, const Tensor* w, float eps, MTL::Buffer* y)
{
    if (dim > 256 || dim % 32 || rot % 2)
        throw std::runtime_error(std::format("heads_partial: heads of {} with {} rotated", dim, rot));
    const HeadsPartialArgs a{ld, col0, stride, dim, rot, n_pad, w ? eps : 0.0f};
    enc->setComputePipelineState(pso("heads_partial"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(cs ? cs : x, 0, 1);
    enc->setBuffer(y, 0, 2);
    enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 3);
    enc->setBytes(&a, sizeof(a), 4);
    enc->dispatchThreadgroups(MTL::Size(T, n_heads, 1), MTL::Size(dim, 1, 1));
    after(enc, "heads_partial");
}

void TransformerRuntime::attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v,
                                   MTL::Buffer* o, uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad,
                                   uint32_t n_heads, uint32_t kv_heads, uint32_t q_pos0, uint32_t block, uint32_t dim)
{
    if (dim != 64 && dim != 128 && dim != 256)
        throw std::runtime_error(std::format("attention: heads of {}", dim));
    const AttnArgs a{n_q, q_pad, n_k, k_pad, n_heads, kv_heads, q_pos0, block, n_heads * dim,
                     1.0f / std::sqrt(float(dim))};
    dispatch(enc, std::format("flash_attn_sg_{}_16x32_4", dim), {q, k, v, o}, &a, sizeof(a),
             MTL::Size(q_pad / kAttnRows, n_heads, 1), MTL::Size(128, 1, 1), false);
}

} // namespace mako::models
