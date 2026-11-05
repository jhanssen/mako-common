#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <map>
#include <string>

namespace mako::mtl {

// y[t][n] = sum_k W[n][k] * x[t][k] for many tokens (prefill), x in half: weights dequantized to half and multiplied
// with matmul2d, float accumulation. F32, Q8_0, Q6_K, Q5_K, Q4_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S and F16 weights; k a
// multiple of 32. BF16 weights take x in bfloat, both read straight from device memory; k a multiple of 64. I8
// weights (int8 rows with a float scale each after the rows, quantize_i8()'s layout) take x the same way, the sums
// exact in int32; k a multiple of 128.
class Gemm {
public:
    explicit Gemm(const Context& ctx);

    static bool supports(gguf::Type type);
    // Up to this many tokens, IQ2_XXS, IQ2_S and IQ3_XXS weights use the few-token kernel (0: never).
    void set_few_tokens(uint32_t t) { few_tokens_ = t; }

    // y rows are ldy elements apart (0: n), x rows ldx (0: k); offsets in bytes.
    // direct: F16 weights read as BF16's are (k a multiple of 64), not staged; the sums come out in another order.
    void encode(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                const MTL::Buffer* w, uint64_t w_off, const MTL::Buffer* xh, uint64_t xh_off, const MTL::Buffer* y,
                uint64_t y_off, uint32_t ldy = 0, uint32_t ldx = 0, bool direct = false);
    // xh = half(x) as dense rows of `cols`; x rows are ldx floats apart (0: cols).
    void to_half(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                 uint32_t ldx, const MTL::Buffer* xh, uint64_t xh_off);
    // The same in bfloat, which BF16 weights take instead (bf16_input()).
    void to_bf16(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                 uint32_t ldx, const MTL::Buffer* xb, uint64_t xb_off);
    static bool bf16_input(gguf::Type type) { return type == gguf::Type::BF16; }
    // q = int8 rows of `cols` of x (float, or bfloat with from_bf16; rows ld elements apart, 0: cols) with a float
    // scale per row after the rows: i8_bytes(), what I8 weights and their x are made of.
    void quantize_i8(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                     uint32_t cols, uint32_t ldx, const MTL::Buffer* q, uint64_t q_off, bool from_bf16 = false);
    static bool i8_input(gguf::Type type) { return type == gguf::Type::I8; }
    static uint64_t i8_bytes(uint64_t rows, uint64_t cols) { return rows * cols + rows * 4; }

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
