// Row invariance of the decode GEMVs: each of T rows (different inputs, as from different sequences) gives the same
// bits as that row run alone, for every weight format, through the plain, multi-GEMV and indexed paths. Random blocks.
//   gemv_rows_test
#include "mtl/context.hpp"
#include "mtl/gemv.hpp"

#include <cstring>
#include <functional>
#include <format>
#include <print>
#include <random>
#include <vector>

namespace {

using mako::gguf::Type;
using mako::mtl::Gemv;
using mako::mtl::Ref;

struct Fixture {
    mako::mtl::Context& ctx;
    Gemv& gemv;
    Ref<MTL::Buffer> table;

    Ref<MTL::Buffer> buffer(uint64_t bytes) const
    {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    }

    template <typename F>
    void run(F&& encode) const
    {
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        encode(enc);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }
};

// Counts outputs of rows run together that differ from the same rows run one at a time.
uint64_t compare(const Fixture& fx, uint32_t rows, uint64_t row_out, MTL::Buffer* together,
                 const std::function<void(MTL::ComputeCommandEncoder*, uint32_t row, const MTL::Buffer* y)>& alone)
{
    uint64_t diff = 0;
    auto y1 = fx.buffer(row_out * 4);
    for (uint32_t r = 0; r < rows; ++r) {
        fx.run([&](MTL::ComputeCommandEncoder* enc) { alone(enc, r, y1.get()); });
        const auto* a = static_cast<const uint32_t*>(together->contents()) + r * row_out;
        const auto* b = static_cast<const uint32_t*>(y1->contents());
        for (uint64_t i = 0; i < row_out; ++i)
            diff += a[i] != b[i];
    }
    return diff;
}

} // namespace

int main()
{
    mako::mtl::Context ctx;
    Gemv gemv(ctx);
    Fixture fx{ctx, gemv, Ref<MTL::Buffer>(ctx.device()->newBuffer(64 * Gemv::kJobBytes, MTL::ResourceStorageModeShared))};
    std::mt19937 rng(23);
    std::normal_distribution<float> nd;
    int failures = 0;

    struct Shape {
        uint32_t k, n;
    };
    const Type types[] = {Type::F32,    Type::Q8_0,   Type::Q6_K,  Type::Q5_K,  Type::Q4_K,   Type::Q3_K,
                          Type::Q2_K,   Type::Q4_0,   Type::IQ4_XS, Type::IQ4_NL, Type::IQ3_S, Type::IQ3_XXS,
                          Type::IQ2_S,  Type::IQ2_XS, Type::IQ2_XXS, Type::IQ1_S, Type::IQ1_M, Type::MXFP4};
    for (const Type type : types) {
        for (const Shape s : {Shape{4096, 1024}, Shape{2304, 1280}, Shape{512, 512}, Shape{256, 384}}) {
            const auto& tr = mako::gguf::traits(type);
            const bool iq = type == Type::IQ3_S || type == Type::IQ3_XXS || type == Type::IQ2_S ||
                            type == Type::IQ2_XS || type == Type::IQ2_XXS || type == Type::IQ1_S || type == Type::IQ1_M;
            if (s.k % tr.block_elems || (iq && s.k < 1024)) // the IQ kernels need rows of 1024 or more
                continue;
            const uint64_t rb = Gemv::row_bytes(type, s.k), mat = rb * s.n;
            const uint32_t n_expert = 8;
            auto w = fx.buffer(mat * n_expert);
            if (type == Type::F32) {
                auto* p = static_cast<float*>(w->contents());
                for (uint64_t i = 0; i < mat * n_expert / 4; ++i)
                    p[i] = nd(rng);
            } else {
                auto* p = static_cast<uint8_t*>(w->contents());
                for (uint64_t i = 0; i < mat * n_expert; ++i)
                    p[i] = uint8_t(rng() & 0x37); // finite f16 scales, every grid and sign index
            }
            for (const uint32_t T : {2u, 3u, 4u, 8u}) {
                auto x = fx.buffer(uint64_t(s.k) * T * 4);
                auto* xf = static_cast<float*>(x->contents());
                for (uint64_t i = 0; i < uint64_t(s.k) * T; ++i)
                    xf[i] = nd(rng);
                std::string line = std::format("{:7} {:4}x{:4} T={}", tr.name, s.k, s.n, T);
                bool ok = true;

                // Plain.
                {
                    auto y = fx.buffer(uint64_t(s.n) * T * 4);
                    Gemv::Args a{type, s.k, s.n, T, 1, false};
                    fx.run([&](MTL::ComputeCommandEncoder* enc) { gemv.encode(enc, a, w.get(), 0, x.get(), 0, y.get(), 0); });
                    const uint64_t d = compare(fx, T, s.n, y.get(), [&](MTL::ComputeCommandEncoder* enc, uint32_t r, const MTL::Buffer* y1) {
                        Gemv::Args a1{type, s.k, s.n, 1, 1, false};
                        gemv.encode(enc, a1, w.get(), 0, x.get(), uint64_t(r) * s.k * 4, y1, 0);
                    });
                    line += std::format("  plain {}", d);
                    ok &= d == 0;
                }
                // Multi-GEMV jobs.
                Gemv::Args aj{type, s.k, s.n, T, 1, false};
                if (gemv.job_supported(aj)) {
                    auto y = fx.buffer(uint64_t(s.n) * T * 4);
                    Gemv::Job j{aj, w.get(), 0, x.get(), 0, y.get(), 0, nullptr, 0, 0, {}, 0};
                    fx.run([&](MTL::ComputeCommandEncoder* enc) { gemv.encode_jobs(enc, std::span(&j, 1), fx.table.get(), 0); });
                    const uint64_t d = compare(fx, T, s.n, y.get(), [&](MTL::ComputeCommandEncoder* enc, uint32_t r, const MTL::Buffer* y1) {
                        Gemv::Job j1{Gemv::Args{type, s.k, s.n, 1, 1, false}, w.get(), 0, x.get(), uint64_t(r) * s.k * 4, y1, 0, nullptr, 0, 0, {}, 0};
                        gemv.encode_jobs(enc, std::span(&j1, 1), fx.table.get(), 0);
                    });
                    line += std::format("  jobs {}", d);
                    ok &= d == 0;
                }
                // Indexed experts: each token its own ids and its own x.
                // MXFP4 and IQ4_NL have no indexed kernel (no expert files use them).
                if (type != Type::F32 && type != Type::MXFP4 && type != Type::IQ4_NL) {
                    constexpr uint32_t sel = 2;
                    auto ids = fx.buffer(uint64_t(T) * sel * 4);
                    auto* id = static_cast<int32_t*>(ids->contents());
                    for (uint32_t i = 0; i < T * sel; ++i)
                        id[i] = int32_t((i * 3 + 1) % n_expert);
                    auto y = fx.buffer(uint64_t(s.n) * sel * T * 4);
                    Gemv::Args a{type, s.k, s.n, T, sel, true};
                    fx.run([&](MTL::ComputeCommandEncoder* enc) {
                        gemv.encode_indexed(enc, a, w.get(), 0, mat, ids.get(), 0, x.get(), 0, y.get(), 0);
                    });
                    const uint64_t d = compare(fx, T, uint64_t(s.n) * sel, y.get(),
                                               [&](MTL::ComputeCommandEncoder* enc, uint32_t r, const MTL::Buffer* y1) {
                        Gemv::Args a1{type, s.k, s.n, 1, sel, true};
                        gemv.encode_indexed(enc, a1, w.get(), 0, mat, ids.get(), uint64_t(r) * sel * 4, x.get(),
                                            uint64_t(r) * s.k * 4, y1, 0);
                    });
                    line += std::format("  indexed {}", d);
                    ok &= d == 0;
                }
                if (!ok) {
                    ++failures;
                    std::println("FAIL {}", line);
                } else if (T == 8) {
                    std::println("ok   {}", line);
                }
            }
        }
    }
    std::println("{}", failures ? std::format("{} FAILED", failures) : "all passed");
    return failures ? 1 : 0;
}
