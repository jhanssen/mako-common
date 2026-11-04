// Flash attention on random half inputs: time and TFLOP/s of each kernel variant, the others compared with the
// first's output. Defaults: MiniMax-H3's sequence at 960x544 (19308 rows, 56 heads of 128, all bidirectional).
//   attn_bench [queries] [keys] [heads] [block] [variant...]
// A variant is NAME:QUERIES_A_THREADGROUP:THREADS (default: the 128-wide kernels with 16 rows a simdgroup).
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using mako::mtl::Ref;

struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t n_q = argc > 1 ? std::stoul(argv[1]) : 19308;
    const uint32_t n_k = argc > 2 ? std::stoul(argv[2]) : n_q;
    const uint32_t heads = argc > 3 ? std::stoul(argv[3]) : 56;
    const uint32_t block = argc > 4 ? std::stoul(argv[4]) : 0;
    std::vector<std::tuple<std::string, uint32_t, uint32_t>> variants;
    for (int i = 5; i < argc; ++i) {
        const std::string v = argv[i];
        const size_t a = v.find(':'), b = v.find(':', a + 1);
        if (a == std::string::npos || b == std::string::npos) {
            std::println(stderr, "variant NAME:QUERIES:THREADS, got {}", v);
            return 2;
        }
        variants.emplace_back(v.substr(0, a), std::stoul(v.substr(a + 1, b - a - 1)), std::stoul(v.substr(b + 1)));
    }
    if (variants.empty())
        variants = {{"flash_attn_sg_128_16x32_4", 64, 128}, {"flash_attn_sg_128_16x64_4", 64, 128}};
    constexpr uint32_t D = 128;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    const uint32_t q_pad = round_up(n_q, 128), k_pad = round_up(n_k, 128);
    auto buffer = [&](uint64_t bytes) {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    auto q = buffer(uint64_t(heads) * q_pad * D * 2), k = buffer(uint64_t(heads) * k_pad * D * 2),
         v = buffer(uint64_t(heads) * k_pad * D * 2);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    for (auto* b : {q.get(), k.get(), v.get()}) {
        auto* h = static_cast<__fp16*>(b->contents());
        for (uint64_t i = 0; i < b->length() / 2; ++i)
            h[i] = __fp16(nd(rng));
    }
    for (uint32_t h = 0; h < heads; ++h) // padded keys and values are zero, as the engines leave them
        for (uint32_t r = n_k; r < k_pad; ++r)
            for (auto* b : {k.get(), v.get()})
                std::memset(static_cast<char*>(b->contents()) + (uint64_t(h) * k_pad + r) * D * 2, 0, D * 2);
    const double flop = 4.0 * n_q * double(n_k) * D * heads;
    std::println("{} queries x {} keys, {} heads of {}: {:.2f} TFLOP", n_q, n_k, heads, D, flop / 1e12);
    std::vector<float> base;
    for (const auto& [name, b, threads] : variants) {
        Ref<MTL::ComputePipelineState> pso;
        try {
            pso = ctx.pipeline(lib.get(), name);
        } catch (const std::exception& e) {
            std::println("{:28} {}", name, e.what());
            continue;
        }
        auto o = buffer(uint64_t(n_q) * heads * D * 4);
        const AttnArgs a{n_q, round_up(n_q, b), n_k, k_pad, heads, heads, 0, block, heads * D,
                         1.0f / std::sqrt(float(D))};
        auto run = [&](int reps) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(q.get(), 0, 0);
            enc->setBuffer(k.get(), 0, 1);
            enc->setBuffer(v.get(), 0, 2);
            enc->setBuffer(o.get(), 0, 3);
            enc->setBytes(&a, sizeof(a), 4);
            for (int i = 0; i < reps; ++i)
                enc->dispatchThreadgroups(MTL::Size(a.q_pad / b, heads, 1), MTL::Size(threads, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / reps;
        };
        run(1);
        const double ms = run(3);
        const auto* of = static_cast<const float*>(o->contents());
        std::string agree = "the reference";
        if (base.empty()) {
            base.assign(of, of + uint64_t(n_q) * heads * D);
        } else {
            double e = 0, r = 0;
            for (size_t i = 0; i < base.size(); ++i) {
                e += (double(of[i]) - base[i]) * (double(of[i]) - base[i]);
                r += double(base[i]) * base[i];
            }
            agree = std::format("rel {:.1e} to the first", std::sqrt(e / r));
        }
        std::println("{:28} {:8.1f} ms  {:5.1f} TFLOP/s  {}", name, ms, flop / ms / 1e9, agree);
    }
    return 0;
}
