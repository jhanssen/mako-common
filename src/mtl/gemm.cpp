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
    case gguf::Type::F16: return "gemm_f16";
    default: return nullptr;
    }
}

} // namespace

Gemm::Gemm(const Context& ctx) : ctx_(ctx), lib_(ctx.load(kernels_metallib())) {}

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
    enc->setComputePipelineState(pso(name));
    enc->setBuffer(w, w_off, 0);
    enc->setBuffer(xh, xh_off, 1);
    enc->setBuffer(y, y_off, 2);
    enc->setBytes(&a, sizeof(a), 3);
    enc->dispatchThreadgroups(MTL::Size((n + 31) / 32, (t + 63) / 64, 1), MTL::Size(128, 1, 1));
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
