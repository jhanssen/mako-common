#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <map>
#include <string>
#include <tuple>

namespace mako::mtl {

// y[b][t][n] = sum_k W[b][n][k] * x[b][t][k] for quantized or F32 weights, fp32 activations.
// Weight batches (b) are consecutive matrices of `n` rows; x and y batches are contiguous.
class Gemv {
public:
    struct Args {
        gguf::Type type;
        uint32_t k = 0;       // input width
        uint32_t n = 0;       // output rows per matrix
        uint32_t t = 1;       // tokens
        uint32_t batch = 1;   // matrices, each with its own y
        bool x_shared = false; // every matrix reads the same x
    };

    explicit Gemv(const Context& ctx);

    static bool supports(gguf::Type type);
    static uint64_t row_bytes(gguf::Type type, uint32_t k);

    void encode(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off);
    // y[i] = W[ids[i]] x_i for i < a.batch (t = 1), with matrix s at w + w_off + s * stride and int32 ids read by
    // the GPU; x_i is x when a.x_shared, else row i of x.
    void encode_indexed(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                        uint64_t stride, const MTL::Buffer* ids, uint64_t ids_off, const MTL::Buffer* x, uint64_t x_off,
                        const MTL::Buffer* y, uint64_t y_off);

private:
    struct Kernel {
        Ref<MTL::ComputePipelineState> pso;
        int nsg = 0, nr0 = 0, smem = 0;
        bool rows_per_sg_grid = false;
    };
    Kernel& kernel(gguf::Type type, uint32_t k, uint32_t batch, bool indexed = false);

    const Context& ctx_;
    Ref<MTL::Library> lib_;
    std::map<std::tuple<gguf::Type, bool, uint32_t, bool>, Kernel> kernels_;
};

} // namespace mako::mtl
