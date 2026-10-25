#include "mtl/gemm.hpp"

#include "mtl/embedded.hpp"

#include <format>
#include <stdexcept>

namespace mako::mtl {

namespace {

// Layout must match GemmArgs in gemm.metal.
struct GemmArgs {
    uint32_t t, k, n, row_bytes, ldy, ldx;
};

const char* kernel_name(gguf::Type type)
{
    switch (type) {
    case gguf::Type::F32: return "gemm_f32";
    case gguf::Type::Q8_0: return "gemm_q8_0";
    case gguf::Type::Q6_K: return "gemm_q6_k";
    case gguf::Type::IQ2_XXS: return "gemm_iq2_xxs";
    case gguf::Type::IQ2_S: return "gemm_iq2_s";
    case gguf::Type::IQ3_XXS: return "gemm_iq3_xxs";
    case gguf::Type::IQ3_S: return "gemm_iq3_s";
    case gguf::Type::F16: return "gemm_f16";
    default: return nullptr;
    }
}

} // namespace

Gemm::Gemm(const Context& ctx) : ctx_(ctx), lib_(ctx.load(kernels_metallib())) {}

const char* few_name(gguf::Type type)
{
    switch (type) {
    case gguf::Type::IQ2_XXS: return "gemm_few_iq2_xxs";
    case gguf::Type::IQ2_S: return "gemm_few_iq2_s";
    case gguf::Type::IQ3_XXS: return "gemm_few_iq3_xxs";
    default: return nullptr;
    }
}

bool Gemm::supports(gguf::Type type) { return kernel_name(type) != nullptr; }

MTL::ComputePipelineState* Gemm::pso(const std::string& name)
{
    auto it = psos_.find(name);
    if (it == psos_.end())
        it = psos_.emplace(name, ctx_.pipeline(lib_.get(), name)).first;
    return it->second.get();
}

void Gemm::encode(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                  const MTL::Buffer* w, uint64_t w_off, const MTL::Buffer* xh, uint64_t xh_off, const MTL::Buffer* y,
                  uint64_t y_off, uint32_t ldy, uint32_t ldx)
{
    const char* name = kernel_name(type);
    if (!name || k % 32)
        throw std::runtime_error(std::format("no GEMM for {} with k={}", gguf::traits(type).name, k));
    const auto tr = gguf::traits(type);
    const GemmArgs a{t, k, n, uint32_t(uint64_t(k) / tr.block_elems * tr.block_bytes), ldy ? ldy : n, ldx ? ldx : k};
    if (const char* few = few_name(type); few && t <= few_tokens_) {
        // Weights dequantized once per pass of up to 4 tokens, in registers; 16 rows per threadgroup.
        const uint32_t nt = t <= 1 ? 1 : t <= 2 ? 2 : 4;
        enc->setComputePipelineState(pso(std::format("{}_{}", few, nt)));
        enc->setBuffer(w, w_off, 0);
        enc->setBuffer(xh, xh_off, 1);
        enc->setBuffer(y, y_off, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
        return;
    }
    // Tiles of 64 tokens, or 32 where that pads less (an expert's share of a chunk is often under 32 tokens).
    uint32_t tm = (t + 31) / 32 * 32 < (t + 63) / 64 * 64 ? 32 : 64, tn = 32;
    const bool iq = type == gguf::Type::IQ2_XXS || type == gguf::Type::IQ2_S || type == gguf::Type::IQ3_XXS ||
                    type == gguf::Type::IQ3_S;
    if (iq && t <= 16)
        tm = 16; // the matrix work of a 32-token tile, not the dequantization, dominates at few tokens
    // 128-token tiles dequantize each weight half as often as 64-token ones; used where they pad no more.
    if (iq && t > 64 && (t + 127) / 128 * 128 <= (t + 63) / 64 * 64)
        tm = 128;
    if (iq && t > 128 && (t + 255) / 256 * 256 <= (t + 127) / 128 * 128)
        tm = 256;
    enc->setComputePipelineState(pso(std::format("{}_{}x{}", name, tm, tn)));
    enc->setBuffer(w, w_off, 0);
    enc->setBuffer(xh, xh_off, 1);
    enc->setBuffer(y, y_off, 2);
    enc->setBytes(&a, sizeof(a), 3);
    enc->dispatchThreadgroups(MTL::Size((n + tn - 1) / tn, (t + tm - 1) / tm, 1), MTL::Size(128, 1, 1));
}

void Gemm::to_half(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                   uint32_t cols, uint32_t ldx, const MTL::Buffer* xh, uint64_t xh_off)
{
    const uint32_t a[2] = {cols, ldx ? ldx : cols};
    enc->setComputePipelineState(pso("to_half"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(xh, xh_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    enc->dispatchThreads(MTL::Size(cols, rows, 1), MTL::Size(256, 1, 1));
}

} // namespace mako::mtl
