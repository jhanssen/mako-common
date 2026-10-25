// Quantizers (Q4_K, Q5_K, Q6_K): round-trip error of random rows through CPU dequantization, and the GPU GEMV and
// prefill GEMM on the quantized rows against the CPU product with the dequantized rows (the layouts the kernels read).
//   quant_test
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"
#include "quant/dequant.hpp"
#include "quant/quantize.hpp"

#include <cmath>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <vector>

int main()
{
    using mako::gguf::Type;
    mako::mtl::Context ctx;
    mako::mtl::Gemv gemv(ctx);
    mako::mtl::Gemm gemm(ctx);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    const uint32_t k = 4096, n = 64;
    // Rows with a few large outliers, as weight rows have.
    std::vector<float> w(uint64_t(k) * n), x(k);
    for (auto& v : w)
        v = nd(rng) * 0.02f * (rng() % 64 == 0 ? 8.0f : 1.0f);
    for (auto& v : x)
        v = nd(rng);
    int failures = 0;
    auto check = [&](bool ok, const std::string& what) {
        std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
        failures += !ok;
    };
    // Expected relative RMS error at these bit widths (Gaussian rows): generous upper bounds.
    const struct {
        Type type;
        double limit;
    } cases[] = {{Type::Q4_K, 0.12}, {Type::Q5_K, 0.06}, {Type::Q6_K, 0.03}};
    for (const auto& c : cases) {
        const uint64_t rb = mako::mtl::Gemv::row_bytes(c.type, k);
        std::vector<std::byte> q(rb * n);
        std::vector<float> back(uint64_t(k) * n);
        for (uint32_t r = 0; r < n; ++r) {
            mako::quant::quantize_row(c.type, w.data() + uint64_t(r) * k, q.data() + r * rb, k);
            mako::quant::dequant_row(c.type, q.data() + r * rb, back.data() + uint64_t(r) * k, k);
        }
        double err = 0, ref = 0;
        for (uint64_t i = 0; i < w.size(); ++i) {
            err += double(back[i] - w[i]) * (back[i] - w[i]);
            ref += double(w[i]) * w[i];
        }
        const double rel = std::sqrt(err / ref);
        check(rel < c.limit, std::format("{}: round trip relative RMS error {:.4f} (limit {})",
                                         mako::gguf::traits(c.type).name, rel, c.limit));

        mako::mtl::Ref<MTL::Buffer> wb(ctx.device()->newBuffer(q.size(), MTL::ResourceStorageModeShared));
        std::memcpy(wb->contents(), q.data(), q.size());
        mako::mtl::Ref<MTL::Buffer> xb(ctx.device()->newBuffer(k * 4, MTL::ResourceStorageModeShared));
        std::memcpy(xb->contents(), x.data(), k * 4);
        mako::mtl::Ref<MTL::Buffer> yb(ctx.device()->newBuffer(n * 4, MTL::ResourceStorageModeShared));
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        gemv.encode(enc, mako::mtl::Gemv::Args{c.type, k, n, 1, 1, false}, wb.get(), 0, xb.get(), 0, yb.get(), 0);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        const auto* y = static_cast<const float*>(yb->contents());
        double gerr = 0, gref = 0;
        for (uint32_t r = 0; r < n; ++r) {
            double dot = 0;
            for (uint32_t i = 0; i < k; ++i)
                dot += double(back[uint64_t(r) * k + i]) * x[i];
            gerr += (y[r] - dot) * (y[r] - dot);
            gref += dot * dot;
        }
        const double grel = std::sqrt(gerr / gref);
        check(grel < 1e-4, std::format("{}: GPU GEMV vs CPU dequantized product, relative error {:.2e}",
                                       mako::gguf::traits(c.type).name, grel));

        // The prefill GEMM (half activations and weights) over T tokens.
        const uint32_t T = 300;
        std::vector<_Float16> xh(uint64_t(T) * k);
        for (auto& v : xh)
            v = _Float16(nd(rng));
        mako::mtl::Ref<MTL::Buffer> xhb(ctx.device()->newBuffer(xh.size() * 2, MTL::ResourceStorageModeShared));
        std::memcpy(xhb->contents(), xh.data(), xh.size() * 2);
        mako::mtl::Ref<MTL::Buffer> ytb(ctx.device()->newBuffer(uint64_t(T) * n * 4, MTL::ResourceStorageModeShared));
        cb = ctx.queue()->commandBuffer();
        enc = cb->computeCommandEncoder();
        gemm.encode(enc, c.type, k, n, T, wb.get(), 0, xhb.get(), 0, ytb.get(), 0);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        const auto* yt = static_cast<const float*>(ytb->contents());
        double merr = 0, mref = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t r = 0; r < n; ++r) {
                double dot = 0;
                for (uint32_t i = 0; i < k; ++i)
                    dot += double(_Float16(back[uint64_t(r) * k + i])) * double(xh[uint64_t(t) * k + i]);
                const double e = yt[uint64_t(t) * n + r] - dot;
                merr += e * e;
                mref += dot * dot;
            }
        const double mrel = std::sqrt(merr / mref);
        check(mrel < 1e-3, std::format("{}: GPU GEMM ({} tokens) vs CPU product with half weights, relative error {:.2e}",
                                       mako::gguf::traits(c.type).name, T, mrel));
    }
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
