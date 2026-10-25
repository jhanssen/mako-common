#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <map>
#include <string>

namespace mako::mtl {

// y[t][n] = sum_k W[n][k] * x[t][k] for many tokens (prefill), x in half: weights dequantized to half and multiplied
// with matmul2d, float accumulation. F32, Q8_0, Q6_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S and F16 weights; k a multiple of 32.
class Gemm {
public:
    explicit Gemm(const Context& ctx);

    static bool supports(gguf::Type type);
    // Up to this many tokens, IQ2_XXS, IQ2_S and IQ3_XXS weights use the few-token kernel (0: never).
    void set_few_tokens(uint32_t t) { few_tokens_ = t; }

    // y rows are ldy elements apart (0: n), x rows ldx (0: k); offsets in bytes.
    void encode(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                const MTL::Buffer* w, uint64_t w_off, const MTL::Buffer* xh, uint64_t xh_off, const MTL::Buffer* y,
                uint64_t y_off, uint32_t ldy = 0, uint32_t ldx = 0);
    // xh = half(x) as dense rows of `cols`; x rows are ldx floats apart (0: cols).
    void to_half(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                 uint32_t ldx, const MTL::Buffer* xh, uint64_t xh_off);

private:
    MTL::ComputePipelineState* pso(const std::string& name);

    const Context& ctx_;
    Ref<MTL::Library> lib_;
    std::map<std::string, Ref<MTL::ComputePipelineState>> psos_;
    // Off: its rounding differs from the tiled kernel's, so a token's result would depend on how many others share
    // its expert (a prompt split differently would no longer give the same logits). Faster up to 2 tokens.
    uint32_t few_tokens_ = 0;
};

} // namespace mako::mtl
