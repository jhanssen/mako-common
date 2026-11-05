// The dense BF16 and int8 GEMMs (weights read straight from device memory) on random inputs: time and TFLOP/s of
// each tile variant, the others compared with the first's output of the same operand type (gemm_i8_... variants get
// int8 rows with a scale each). Defaults: MiniMax-H3's FFN down projection at 960x544 (19456 rows x 14336 -> 5376).
//   dense_gemm_bench [rows] [k] [n] [variant...]      (a variant: NAME:THREADS, e.g. gemm_bf16_256x64:256)
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

using mako::mtl::Ref;

struct GemmArgs {
    uint32_t t, k, n, row_bytes, ldy, ldx;
};

uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const uint32_t t = argc > 1 ? std::stoul(argv[1]) : 19456;
    const uint32_t k = argc > 2 ? std::stoul(argv[2]) : 14336;
    const uint32_t n = argc > 3 ? std::stoul(argv[3]) : 5376;
    std::vector<std::pair<std::string, uint32_t>> variants;
    for (int i = 4; i < argc; ++i) {
        const std::string v = argv[i];
        const size_t a = v.find(':');
        variants.emplace_back(v.substr(0, a), a == std::string::npos ? 128 : std::stoul(v.substr(a + 1)));
    }
    if (variants.empty())
        variants = {{"gemm_bf16_128x64", 128}, {"gemm_bf16_64x128", 128}};
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto buffer = [&](uint64_t bytes) {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    auto w = buffer(uint64_t(n) * k * 2), x = buffer(uint64_t(t) * k * 2);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    for (auto* b : {w.get(), x.get()}) {
        auto* h = static_cast<uint16_t*>(b->contents());
        for (uint64_t i = 0; i < b->length() / 2; ++i)
            h[i] = to_bf16(nd(rng) * 0.05f);
    }
    // Int8 operands: rows of random int8 with a scale each after the rows.
    auto wq = buffer(uint64_t(n) * k + uint64_t(n) * 4), xq = buffer(uint64_t(t) * k + uint64_t(t) * 4);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (auto [b, rows] : {std::pair{wq.get(), n}, std::pair{xq.get(), t}}) {
        auto* q = static_cast<int8_t*>(b->contents());
        for (uint64_t i = 0; i < uint64_t(rows) * k; ++i)
            q[i] = int8_t(qd(rng));
        auto* sc = reinterpret_cast<float*>(q + uint64_t(rows) * k);
        for (uint32_t r = 0; r < rows; ++r)
            sc[r] = 0.05f / 127.0f * (0.5f + float(r % 7) / 7.0f);
    }
    const double flop = 2.0 * t * double(k) * n;
    std::println("{} rows x {} -> {}: {:.2f} TFLOP, weights {:.0f} MB, activations {:.0f} MB", t, k, n, flop / 1e12,
                 double(n) * k * 2 / 1e6, double(t) * k * 2 / 1e6);
    std::vector<float> base, base_i8;
    for (const auto& [name, threads] : variants) {
        const bool i8 = name.starts_with("gemm_i8");
        Ref<MTL::ComputePipelineState> pso;
        try {
            pso = ctx.pipeline(lib.get(), name);
        } catch (const std::exception& e) {
            std::println("{:20} {}", name, e.what());
            continue;
        }
        // The tile from the name: NAME_<tm>x<tn>.
        const size_t us = name.rfind('_'), xx = name.rfind('x');
        const uint32_t tm = std::stoul(name.substr(us + 1, xx - us - 1)), tn = std::stoul(name.substr(xx + 1));
        auto y = buffer(uint64_t(t) * n * 4);
        const GemmArgs a{t, k, n, i8 ? k : k * 2, n, k};
        auto run = [&](int reps) {
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(i8 ? wq.get() : w.get(), 0, 0);
            enc->setBuffer(i8 ? xq.get() : x.get(), 0, 1);
            enc->setBuffer(y.get(), 0, 2);
            enc->setBytes(&a, sizeof(a), 3);
            for (int i = 0; i < reps; ++i)
                enc->dispatchThreadgroups(MTL::Size((n + tn - 1) / tn, (t + tm - 1) / tm, 1), MTL::Size(threads, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / reps;
        };
        run(1);
        const double ms = run(5);
        const auto* yf = static_cast<const float*>(y->contents());
        std::string agree = i8 ? "the int8 reference" : "the reference";
        std::vector<float>& ref = i8 ? base_i8 : base;
        if (ref.empty()) {
            ref.assign(yf, yf + uint64_t(t) * n);
        } else {
            double e = 0, r = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                e += (double(yf[i]) - ref[i]) * (double(yf[i]) - ref[i]);
                r += double(ref[i]) * ref[i];
            }
            agree = std::format("rel {:.1e} to the first{}", std::sqrt(e / r), i8 ? " int8" : "");
        }
        std::println("{:20} {:8.2f} ms  {:5.1f} TFLOP/s  {}", name, ms, flop / ms / 1e9, agree);
    }
    return 0;
}
