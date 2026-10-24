#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <map>
#include <string>

namespace mako::mtl {

// y[t][n] = sum_k W[n][k] * x[t][k] for many tokens (prefill): tiles of weights dequantized to half and multiplied
// with matmul2d, float accumulation. F32, Q8_0 and Q6_K weights; k a multiple of 64.
class Gemm {
public:
    explicit Gemm(const Context& ctx);

    static bool supports(gguf::Type type);

    void encode(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                const MTL::Buffer* w, uint64_t w_off, const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y,
                uint64_t y_off);

private:
    const Context& ctx_;
    Ref<MTL::Library> lib_;
    std::map<std::string, Ref<MTL::ComputePipelineState>> psos_;
};

} // namespace mako::mtl
