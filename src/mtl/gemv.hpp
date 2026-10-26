#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"

#include <functional>
#include <map>
#include <span>
#include <string>
#include <tuple>

namespace mako::mtl {

// y[b][t][n] = sum_k W[b][n][k] * x[b][t][k] for quantized or F32 weights, fp32 activations.
// Weight batches (b) are consecutive matrices of `n` rows; x and y batches are contiguous.
// One F32 or Q8_0 matrix with short rows (k <= 512) goes to mako's own kernel; the per-format kernels give each
// row 32 lanes and leave most of them idle there.
struct GemvTail {
    const MTL::Buffer* buf = nullptr;
    uint64_t off = 0;
    uint32_t split = UINT32_MAX;
};

class Gemv {
public:
    struct Args {
        gguf::Type type;
        uint32_t k = 0;       // input width
        uint32_t n = 0;       // output rows per matrix
        uint32_t t = 1;       // tokens
        uint32_t batch = 1;   // matrices, each with its own y
        bool x_shared = false; // every matrix reads the same x
        uint32_t ids_stride = 0; // encode_indexed: bytes between tokens' id rows (0: batch ids, packed)
    };

    explicit Gemv(const Context& ctx);

    static bool supports(gguf::Type type);
    static uint64_t row_bytes(gguf::Type type, uint32_t k);

    // The F32 router GEMV that also routes each token in its last threadgroup (mako_mul_mv_f32_f32_4_route):
    // bind_route binds the routing's buffers (4..14) before the dispatch. False (nothing encoded) when the shape is
    // not the kernel's: F32, k a multiple of 4, one matrix, n <= 512, up to 8 tokens.
    bool encode_router(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                       const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off,
                       const std::function<void(MTL::ComputeCommandEncoder*)>& bind_route);
    void encode(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off);

    // One GEMV of a multi-GEMV dispatch (encode_jobs): as encode() (ids null) or encode_indexed() (ids set) would
    // run it, the outputs laid out the same way.
    struct Job {
        Args a;
        const MTL::Buffer* w = nullptr;
        uint64_t w_off = 0;
        const MTL::Buffer* x = nullptr;
        uint64_t x_off = 0;
        const MTL::Buffer* y = nullptr;
        uint64_t y_off = 0;
        const MTL::Buffer* ids = nullptr; // indexed: int32 slot ids, a row per token
        uint64_t ids_off = 0, stride = 0; // indexed: bytes between slots
        GemvTail tail;
        uint32_t ldy = 0; // plain jobs: y row stride in elements (0: n)
    };
    // Whether encode_jobs runs this GEMV: Q6_K, Q8_0 and F32 rows of >= 512 (no short-row kernel), IQ2_XXS,
    // IQ3_XXS and IQ2_S, up to 8 tokens.
    bool job_supported(const Args& a) const;
    // Independent GEMVs over the same tokens in one dispatch (mako_mul_mv_jobs): one ramp-up and tail instead of
    // one per GEMV. The job table (kJobBytes a job) is written at table + table_off, which the GPU reads.
    static constexpr uint64_t kJobBytes = 128;
    void encode_jobs(MTL::ComputeCommandEncoder* enc, std::span<const Job> jobs, MTL::Buffer* table,
                     uint64_t table_off);
    // y[i] = W[ids[i]] x_i for i < a.batch (t = 1), with matrix s at w + w_off + s * stride and int32 ids read by
    // the GPU; x_i is x when a.x_shared, else row i of x. With `indirect`, the threadgroup grid is read from it
    // (MTLDispatchThreadgroupsIndirectArguments): x and y as indexed_grid() gives them, z the number of ids to run.
    // Matrices s >= tail.split are at tail.buf + tail.off + (s - split) * stride instead; with tail.buf null (the
    // arena's lendable slots lent) no id may reach split.
    using Tail = GemvTail;
    void encode_indexed(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                        uint64_t stride, const MTL::Buffer* ids, uint64_t ids_off, const MTL::Buffer* x, uint64_t x_off,
                        const MTL::Buffer* y, uint64_t y_off, const MTL::Buffer* indirect = nullptr,
                        uint64_t indirect_off = 0, const Tail& tail = {});
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
    // tokens > 1: a kernel that runs that many tokens per threadgroup (weights read once), where there is one.
    Kernel& kernel(gguf::Type type, uint32_t k, uint32_t batch, bool indexed = false, uint32_t tokens = 1);

    bool encode_short(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                      const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off);
    MTL::ComputePipelineState* own(const std::string& name);

    const Context& ctx_;
    Ref<MTL::Library> lib_, own_lib_;
    std::map<std::string, Ref<MTL::ComputePipelineState>> own_psos_;
    bool own_kernels_ = true;
    Ref<MTL::ComputePipelineState> router_pso_; // encode_router: built for the first k seen
    int router_nsg_ = 0;
    Ref<MTL::ComputePipelineState> jobs_pso_;
    std::map<std::tuple<gguf::Type, bool, uint32_t, bool, uint32_t>, Kernel> kernels_;
};

} // namespace mako::mtl
