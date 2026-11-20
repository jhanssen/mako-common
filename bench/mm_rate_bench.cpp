// matmul2d's rate on attention-sized tiles (bench/attn_ct.metal's mm_rate kernels): independent single-simdgroup
// products against cooperative ones over a threadgroup.
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <print>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct RateArgs {
    uint32_t iters, k_pad;
};

} // namespace

int main()
{
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    constexpr uint32_t D = 128, k_pad = 19328, groups = 4096, iters = 302; // 302 x 64 keys ~ 19k
    auto buffer = [&](uint64_t bytes) {
        return mako::mtl::Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    auto q = buffer(uint64_t(groups) * 64 * D * 2), k = buffer(uint64_t(k_pad) * D * 2),
         o = buffer(uint64_t(groups) * 64 * 128 * 4);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    for (auto* b : {q.get(), k.get()}) {
        auto* h = static_cast<__fp16*>(b->contents());
        for (uint64_t i = 0; i < b->length() / 2; ++i)
            h[i] = __fp16(nd(rng));
    }
    // name, rows a threadgroup (M x NSG for independent, M for cooperative), N, threads; the first two repeat at the
    // end because the first kernel of a run measures 49-65 TFLOP/s (clock ramp), the later ones within 1%.
    for (const auto& [name, m, n, nsg, coop] :
         std::vector<std::tuple<std::string, uint32_t, uint32_t, uint32_t, bool>>{
             {"mm_rate_16x64x128_sg", 16, 64, 4, false},     {"mm_rate_16x128x128_sg", 16, 128, 4, false},
             {"mm_rate_32x64x128_sg", 32, 64, 4, false},     {"mm_rate_64x64x128_coop4", 64, 64, 4, true},
             {"mm_rate_64x128x128_coop4", 64, 128, 4, true}, {"mm_rate_32x64x128_coop2", 32, 64, 2, true},
             {"mm_rate_bf16_64x128x128_coop4", 64, 128, 4, true}, {"mm_rate_i8_64x128x128_coop4", 64, 128, 4, true},
             {"mm_rate_i8_64x64x128_coop4", 64, 64, 4, true}, {"mm_rate_bf16_128x128x128_coop8", 128, 128, 8, true},
             {"mm_rate_f16_128x128x128_coop8", 128, 128, 8, true}, {"mm_rate_i8_128x128x128_coop8", 128, 128, 8, true},
             {"mm_rate_16x64x128_sg", 16, 64, 4, false}, {"mm_rate_16x128x128_sg", 16, 128, 4, false}}) {
        auto pso = ctx.pipeline(lib.get(), name);
        const uint32_t rows_tg = coop ? m : m * nsg, tgs = groups * 64 / rows_tg;
        const RateArgs a{iters, k_pad};
        auto run = [&](int reps) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(q.get(), 0, 0);
            enc->setBuffer(k.get(), 0, 1);
            enc->setBuffer(o.get(), 0, 2);
            enc->setBytes(&a, sizeof(a), 3);
            for (int i = 0; i < reps; ++i)
                enc->dispatchThreadgroups(MTL::Size(tgs, 1, 1), MTL::Size(32 * nsg, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / reps;
        };
        run(1);
        const double ms = run(3);
        const double flop = 2.0 * groups * 64 * double(n) * D * iters;
        std::println("{:28} {:8.1f} ms  {:5.1f} TFLOP/s", name, ms, flop / ms / 1e9);
    }
    return 0;
}
