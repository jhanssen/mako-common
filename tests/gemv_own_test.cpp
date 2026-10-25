// mako's own GEMV kernels against the per-format ones they replace, bit for bit, on random blocks: plain (one x),
// a batch sharing x, and indexed by expert ids.
//   gemv_own_test
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <vector>

int main()
{
    using mako::gguf::Type;
    mako::mtl::Context ctx;
    mako::mtl::Gemv own(ctx), ref(ctx);
    ref.set_own_kernels(false);
    std::mt19937 rng(17);
    std::normal_distribution<float> nd;
    int failures = 0;
    struct Case {
        Type type;
        uint32_t k, n;
    };
    for (const Case c : {Case{Type::IQ2_XXS, 5120, 2304}, Case{Type::IQ2_XXS, 4096, 1000}, Case{Type::IQ2_XXS, 2048, 4096}}) {
        const uint32_t batch = 6, n_expert = 16;
        const uint64_t rb = mako::mtl::Gemv::row_bytes(c.type, c.k), mat = rb * c.n;
        mako::mtl::Ref<MTL::Buffer> w(ctx.device()->newBuffer(mat * n_expert, MTL::ResourceStorageModeShared));
        auto* p = static_cast<uint8_t*>(w->contents());
        for (uint64_t i = 0; i < mat * n_expert; ++i)
            p[i] = uint8_t(rng() & 0x37); // finite f16 scales, every grid and sign index
        mako::mtl::Ref<MTL::Buffer> x(ctx.device()->newBuffer(uint64_t(c.k) * batch * 4, MTL::ResourceStorageModeShared));
        auto* xf = static_cast<float*>(x->contents());
        for (uint64_t i = 0; i < uint64_t(c.k) * batch; ++i)
            xf[i] = nd(rng);
        mako::mtl::Ref<MTL::Buffer> ids(ctx.device()->newBuffer(64, MTL::ResourceStorageModeShared));
        auto* id = static_cast<int32_t*>(ids->contents());
        for (uint32_t i = 0; i < batch; ++i)
            id[i] = int32_t((i * 5 + 3) % n_expert);
        const uint64_t ybytes = uint64_t(c.n) * batch * 4;
        auto run = [&](mako::mtl::Gemv& g, bool indexed) {
            mako::mtl::Ref<MTL::Buffer> y(ctx.device()->newBuffer(ybytes, MTL::ResourceStorageModeShared));
            auto* cb = ctx.queue()->commandBuffer();
            auto* enc = cb->computeCommandEncoder();
            mako::mtl::Gemv::Args a{c.type, c.k, c.n, 1, batch, true};
            if (indexed)
                g.encode_indexed(enc, a, w.get(), 0, mat, ids.get(), 0, x.get(), 0, y.get(), 0);
            else
                g.encode(enc, a, w.get(), 0, x.get(), 0, y.get(), 0);
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            std::vector<float> out(uint64_t(c.n) * batch);
            std::memcpy(out.data(), y->contents(), ybytes);
            return out;
        };
        for (const bool indexed : {false, true}) {
            const auto a = run(own, indexed), b = run(ref, indexed);
            uint64_t diff = 0;
            double err = 0, ref2 = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                diff += std::memcmp(&a[i], &b[i], 4) != 0;
                err += double(a[i] - b[i]) * (a[i] - b[i]);
                ref2 += double(b[i]) * b[i];
            }
            const bool ok = diff == 0;
            std::println("  {}  {} {}x{} {}: {} of {} outputs differ (relative RMS {:.1e})", ok ? "ok  " : "FAIL",
                         mako::gguf::traits(c.type).name, c.k, c.n, indexed ? "indexed" : "batch", diff, a.size(),
                         std::sqrt(err / ref2));
            failures += !ok;
            if (!ok && std::getenv("GEMV_OWN_DEBUG"))
                for (int i = 0; i < 4; ++i)
                    std::println("      [{}] own {} ref {}", i, a[i], b[i]);
        }
    }
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
