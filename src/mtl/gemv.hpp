#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <map>
#include <string>
#include <tuple>

namespace mako::mtl {

// y[b][t][n] = sum_k W[b][n][k] * x[b][t][k] for quantized or F32 weights, fp32 activations.
// Weight batches (b) are consecutive matrices of `n` rows; x and y batches are contiguous.
// One F32 or Q8_0 matrix with short rows (k <= 512) goes to mako's own kernel; the per-format kernels give each
// row 32 lanes and leave most of them idle there.
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
    // the GPU; x_i is x when a.x_shared, else row i of x. With `indirect`, the threadgroup grid is read from it
    // (MTLDispatchThreadgroupsIndirectArguments): x and y as indexed_grid() gives them, z the number of ids to run.
    void encode_indexed(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                        uint64_t stride, const MTL::Buffer* ids, uint64_t ids_off, const MTL::Buffer* x, uint64_t x_off,
                        const MTL::Buffer* y, uint64_t y_off, const MTL::Buffer* indirect = nullptr,
                        uint64_t indirect_off = 0);
    // Off: always the per-format kernels (for comparisons).
    void set_own_kernels(bool on) { own_kernels_ = on; }
    // The threadgroup grid encode_indexed dispatches for all a.batch ids.
    MTL::Size indexed_grid(const Args& a);

private:
    struct Kernel {
        Ref<MTL::ComputePipelineState> pso;
        int nsg = 0, nr0 = 0, smem = 0;
        bool rows_per_sg_grid = false;
    };
    Kernel& kernel(gguf::Type type, uint32_t k, uint32_t batch, bool indexed = false);

    bool encode_short(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                      const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off);
    MTL::ComputePipelineState* own(const std::string& name);

    const Context& ctx_;
    Ref<MTL::Library> lib_, own_lib_;
    std::map<std::string, Ref<MTL::ComputePipelineState>> own_psos_;
    bool own_kernels_ = true;
    std::map<std::tuple<gguf::Type, bool, uint32_t, bool>, Kernel> kernels_;
};

} // namespace mako::mtl
