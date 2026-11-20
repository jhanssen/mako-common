// The int8 GEMM path against CPU references: quant_rows_i8 over float and bfloat rows, gemm_i8 on full and partial
// tiles (the int32 sums exact, so the outputs match to float rounding), lora_merge_i8 against a float merge
// requantized on the CPU.
//   int8_test
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"
#include "mtl/gemm.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <vector>

namespace {

using mako::mtl::Ref;

struct LoraArgs {
    uint32_t k, rows, rank, b_rot;
    float scale;
};

uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float from_bf16(uint16_t b)
{
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// The kernels' quantization of one row: s = max |x| / 127 (0: 1 / 127), q = clamp(rint(x / s)).
void quantize_row(const float* x, uint32_t n, int8_t* q, float& scale)
{
    float m = 0.0f;
    for (uint32_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(x[i]));
    scale = m > 0.0f ? m / 127.0f : 1.0f / 127.0f;
    const float inv = 1.0f / scale;
    for (uint32_t i = 0; i < n; ++i)
        q[i] = int8_t(std::clamp(std::rint(x[i] * inv), -127.0f, 127.0f));
}

int failures = 0;
void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    mako::mtl::Context ctx;
    mako::mtl::Gemm gemm(ctx);
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto buffer = [&](uint64_t bytes) {
        return Ref<MTL::Buffer>(ctx.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
    };
    auto run = [&](auto&& encode) {
        auto* cb = ctx.queue()->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        encode(enc);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    };
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;

    // Quantization of float rows (one of them all zero) and of bfloat rows.
    for (const bool bf16 : {false, true}) {
        const uint32_t rows = bf16 ? 70 : 37, cols = bf16 ? 256 : 512, ld = cols + 8;
        std::vector<float> x(uint64_t(rows) * ld);
        for (auto& v : x)
            v = nd(rng) * 3.0f;
        std::fill_n(x.begin() + 5 * ld, cols, 0.0f);
        auto xb = buffer(x.size() * (bf16 ? 2 : 4));
        if (bf16) {
            auto* h = static_cast<uint16_t*>(xb->contents());
            for (size_t i = 0; i < x.size(); ++i) {
                h[i] = to_bf16(x[i]);
                x[i] = from_bf16(h[i]); // the reference sees what the kernel sees
            }
        } else {
            std::memcpy(xb->contents(), x.data(), x.size() * 4);
        }
        auto q = buffer(mako::mtl::Gemm::i8_bytes(rows, cols));
        run([&](MTL::ComputeCommandEncoder* enc) { gemm.quantize_i8(enc, xb.get(), 0, rows, cols, ld, q.get(), 0, bf16); });
        const auto* gq = static_cast<const int8_t*>(q->contents());
        const auto* gs = reinterpret_cast<const float*>(gq + uint64_t(rows) * cols);
        // The kernel's max / 127 is fast-math division (a reciprocal), so a scale can be an ulp off the CPU's and a
        // value at a rounding boundary off by one.
        uint64_t off_by_one = 0, worse = 0;
        double scale_err = 0;
        std::vector<int8_t> rq(cols);
        for (uint32_t r = 0; r < rows; ++r) {
            float s;
            quantize_row(&x[uint64_t(r) * ld], cols, rq.data(), s);
            scale_err = std::max(scale_err, double(std::fabs(gs[r] - s)) / s);
            for (uint32_t c = 0; c < cols; ++c) {
                const int d = std::abs(int(gq[uint64_t(r) * cols + c]) - int(rq[c]));
                off_by_one += d == 1;
                worse += d > 1;
            }
        }
        check(scale_err < 2e-7 && worse == 0 && off_by_one < uint64_t(rows) * cols / 500,
              std::format("quant_rows_i8 {}: scales within {:.1e}, {} values off by one, {} by more", bf16 ? "bf16" : "f32",
                          scale_err, off_by_one, worse));
    }

    // The GEMM on tile-aligned and ragged shapes, dense and strided output.
    for (const auto [t, k, n, ldy] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>{
             {70, 256, 200, 0}, {200, 384, 256, 0}, {33, 128, 64, 0}, {130, 256, 300, 364}, {16, 128, 128, 0}}) {
        auto wq = buffer(mako::mtl::Gemm::i8_bytes(n, k)), xq = buffer(mako::mtl::Gemm::i8_bytes(t, k));
        std::uniform_int_distribution<int> qd(-127, 127);
        std::uniform_real_distribution<float> sd(0.001f, 0.1f);
        for (auto [b, rows] : {std::pair{wq.get(), n}, std::pair{xq.get(), t}}) {
            auto* q = static_cast<int8_t*>(b->contents());
            for (uint64_t i = 0; i < uint64_t(rows) * k; ++i)
                q[i] = int8_t(qd(rng));
            auto* sc = reinterpret_cast<float*>(q + uint64_t(rows) * k);
            for (uint32_t r = 0; r < rows; ++r)
                sc[r] = sd(rng);
        }
        const uint32_t ly = ldy ? ldy : n;
        auto y = buffer(uint64_t(t) * ly * 4);
        std::memset(y->contents(), 0, y->length());
        run([&](MTL::ComputeCommandEncoder* enc) {
            gemm.encode(enc, mako::gguf::Type::I8, k, n, t, wq.get(), 0, xq.get(), 0, y.get(), 0, ldy);
        });
        const auto* w8 = static_cast<const int8_t*>(wq->contents());
        const auto* x8 = static_cast<const int8_t*>(xq->contents());
        const auto* ws = reinterpret_cast<const float*>(w8 + uint64_t(n) * k);
        const auto* xs = reinterpret_cast<const float*>(x8 + uint64_t(t) * k);
        const auto* yf = static_cast<const float*>(y->contents());
        double worst = 0;
        uint64_t touched = 0;
        for (uint32_t r = 0; r < t; ++r)
            for (uint32_t c = 0; c < ly; ++c) {
                const float got = yf[uint64_t(r) * ly + c];
                if (c >= n) {
                    touched += got != 0.0f;
                    continue;
                }
                int32_t acc = 0;
                for (uint32_t i = 0; i < k; ++i)
                    acc += int32_t(x8[uint64_t(r) * k + i]) * int32_t(w8[uint64_t(c) * k + i]);
                const float want = float(acc) * xs[r] * ws[c];
                worst = std::max(worst, double(std::fabs(got - want)) / std::max(1e-6f, std::fabs(want)));
            }
        check(worst < 2e-6 && touched == 0, std::format("gemm_i8 {}x{}->{} ldy {}: worst rel {:.1e}, {} pad values touched", t, k, n, ly, worst, touched));
    }

    // The LoRA merge into int8 rows: W' = requantize(s q + scale B A) with B's rows rotated.
    {
        const uint32_t rows = 96, k = 512, rank = 8, rot = 32;
        const float scale = 0.7f;
        std::vector<float> w(uint64_t(rows) * k), a(uint64_t(rank) * k), b(uint64_t(rows) * rank);
        for (auto& v : w)
            v = nd(rng);
        for (auto& v : a)
            v = nd(rng) * 0.1f;
        for (auto& v : b)
            v = nd(rng) * 0.1f;
        auto wq = buffer(mako::mtl::Gemm::i8_bytes(rows, k));
        auto* q = static_cast<int8_t*>(wq->contents());
        auto* sc = reinterpret_cast<float*>(q + uint64_t(rows) * k);
        std::vector<float> ref(uint64_t(rows) * k);
        for (uint32_t r = 0; r < rows; ++r) {
            quantize_row(&w[uint64_t(r) * k], k, q + uint64_t(r) * k, sc[r]);
            for (uint32_t c = 0; c < k; ++c) {
                float d = 0.0f;
                for (uint32_t j = 0; j < rank; ++j)
                    d += b[uint64_t((r + rot) % rows) * rank + j] * a[uint64_t(j) * k + c];
                ref[uint64_t(r) * k + c] = sc[r] * float(q[uint64_t(r) * k + c]) + scale * d;
            }
        }
        auto ab = buffer(a.size() * 4), bb = buffer(b.size() * 4);
        std::memcpy(ab->contents(), a.data(), a.size() * 4);
        std::memcpy(bb->contents(), b.data(), b.size() * 4);
        auto pso = ctx.pipeline(lib.get(), "lora_merge_i8");
        const LoraArgs la{k, rows, rank, rot, scale};
        run([&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(pso.get());
            enc->setBuffer(wq.get(), 0, 0);
            enc->setBuffer(ab.get(), 0, 1);
            enc->setBuffer(bb.get(), 0, 2);
            enc->setBytes(&la, sizeof(la), 3);
            enc->setBuffer(wq.get(), uint64_t(rows) * k, 4);
            enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        });
        uint64_t off_by_one = 0, worse = 0;
        double scale_err = 0;
        std::vector<int8_t> rq(k);
        for (uint32_t r = 0; r < rows; ++r) {
            float s;
            quantize_row(&ref[uint64_t(r) * k], k, rq.data(), s);
            scale_err = std::max(scale_err, double(std::fabs(sc[r] - s)) / s);
            for (uint32_t c = 0; c < k; ++c) {
                const int d = std::abs(int(q[uint64_t(r) * k + c]) - int(rq[c]));
                off_by_one += d == 1;
                worse += d > 1;
            }
        }
        check(scale_err < 1e-5 && worse == 0 && off_by_one < uint64_t(rows) * k / 200,
              std::format("lora_merge_i8: scales within {:.1e}, {} values off by one (fma order), {} by more", scale_err, off_by_one, worse));
    }
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
