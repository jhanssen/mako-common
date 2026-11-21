#include "mtl/gemm.hpp"

#include "mtl/embedded.hpp"

#include <array>
#include <format>
#include <stdexcept>

namespace mako::mtl {

namespace {

// Layout must match GemmArgs in gemm.metal.
struct GemmArgs {
    uint32_t t, k, n, row_bytes, ldy, ldx;
    uint32_t batch = 1, t_tiles = 0, w_mat = 0, x_mat = 0, y_mat = 0;
};

const char* kernel_name(gguf::Type type)
{
    switch (type) {
    case gguf::Type::F32: return "gemm_f32";
    case gguf::Type::Q8_0: return "gemm_q8_0";
    case gguf::Type::Q6_K: return "gemm_q6_k";
    case gguf::Type::Q4_K: return "gemm_q4_k";
    case gguf::Type::Q5_K: return "gemm_q5_k";
    case gguf::Type::IQ2_XXS: return "gemm_iq2_xxs";
    case gguf::Type::IQ2_S: return "gemm_iq2_s";
    case gguf::Type::IQ3_XXS: return "gemm_iq3_xxs";
    case gguf::Type::IQ3_S: return "gemm_iq3_s";
    case gguf::Type::IQ4_XS: return "gemm_iq4_xs";
    case gguf::Type::IQ2_XS: return "gemm_iq2_xs";
    case gguf::Type::IQ1_S: return "gemm_iq1_s";
    case gguf::Type::IQ1_M: return "gemm_iq1_m";
    case gguf::Type::F16: return "gemm_f16";
    case gguf::Type::BF16: return "gemm_bf16";
    case gguf::Type::I8: return "gemm_i8";
    default: return nullptr;
    }
}

} // namespace

Gemm::Gemm(const Context& ctx) : ctx_(ctx), lib_(ctx.load(kernels_metallib())) {}

const char* few_name(gguf::Type type)
{
    switch (type) {
    case gguf::Type::IQ2_XXS: return "gemm_few_iq2_xxs";
    case gguf::Type::IQ2_S: return "gemm_few_iq2_s";
    case gguf::Type::IQ3_XXS: return "gemm_few_iq3_xxs";
    default: return nullptr;
    }
}

bool Gemm::supports(gguf::Type type) { return kernel_name(type) != nullptr; }

MTL::ComputePipelineState* Gemm::pso(const std::string& name)
{
    auto it = psos_.find(name);
    if (it == psos_.end())
        it = psos_.emplace(name, ctx_.pipeline(lib_.get(), name)).first;
    return it->second.get();
}

namespace {

bool is_iq(gguf::Type type)
{
    return type == gguf::Type::IQ2_XXS || type == gguf::Type::IQ2_XS || type == gguf::Type::IQ2_S ||
           type == gguf::Type::IQ3_XXS || type == gguf::Type::IQ3_S || type == gguf::Type::IQ1_S ||
           type == gguf::Type::IQ1_M;
}

// The staged kernels' token tile for t tokens.
uint32_t staged_tile(gguf::Type type, uint32_t t)
{
    // Tiles of 64 tokens, or 32 where that pads less (an expert's share of a chunk is often under 32 tokens).
    uint32_t tm = (t + 31) / 32 * 32 < (t + 63) / 64 * 64 ? 32 : 64;
    if (is_iq(type) && t <= 16)
        tm = 16; // the matrix work of a 32-token tile, not the dequantization, dominates at few tokens
    // 128- and 256-token tiles dequantize each weight half / a quarter as often as 64-token ones; used where they pad
    // no more.
    if (t > 64 && (t + 127) / 128 * 128 <= (t + 63) / 64 * 64)
        tm = 128;
    if (t > 128 && (t + 255) / 256 * 256 <= (t + 127) / 128 * 128)
        tm = 256;
    return tm;
}

// Runs of `count` tiles of `tm` tokens, together the t tokens, in token order; only the last tile can be partial.
struct TileRun {
    uint32_t tm, count;
};

// The staged kernels' tiles for t tokens, the set of tile sizes costing least: one tile's time relative to a
// 128-token one's, measured on IQ3_XXS 1536 x 1536 experts (16: 0.33, 32: 0.45, 64: 0.65, 256: 1.86). 160 tokens,
// say, run as 128 + 32 rather than as one 256-token tile.
std::vector<TileRun> plan_tiles(gguf::Type type, uint32_t t)
{
    constexpr uint32_t sizes[] = {16, 32, 64, 128, 256};
    constexpr float cost[] = {0.33f, 0.45f, 0.65f, 1.0f, 1.86f};
    const uint32_t first = is_iq(type) ? 0 : 1; // 16-token tiles: IQ weights only
    std::vector<TileRun> runs;
    const uint32_t big = t > 512 ? (t - 256) / 256 : 0; // whole 256-token tiles before the search
    if (big)
        runs.push_back({256, big});
    const uint32_t rest = t - big * 256; // at most 767
    std::array<float, 768> best;
    std::array<uint8_t, 768> pick;
    best[0] = 0.0f;
    for (uint32_t r = 1; r <= rest; ++r) {
        best[r] = 1e30f;
        for (uint32_t i = first; i < 5; ++i) {
            const float c = best[r > sizes[i] ? r - sizes[i] : 0] + cost[i];
            if (c < best[r]) {
                best[r] = c;
                pick[r] = uint8_t(i);
            }
        }
    }
    // Whole tiles by size, largest first, then the run holding the partial one (the search's last step) last.
    uint32_t whole[5] = {}, partial = 5;
    for (uint32_t r = rest; r > 0;) {
        const uint32_t i = pick[r];
        if (r < sizes[i])
            partial = i;
        else
            ++whole[i];
        r = r > sizes[i] ? r - sizes[i] : 0;
    }
    for (uint32_t i = 5; i-- > 0;)
        if (i != partial && whole[i]) {
            if (!runs.empty() && runs.back().tm == sizes[i])
                runs.back().count += whole[i];
            else
                runs.push_back({sizes[i], whole[i]});
        }
    if (partial < 5)
        runs.push_back({sizes[partial], whole[partial] + 1});
    return runs;
}

const char* glu_name(gguf::Type type)
{
    switch (type) {
    case gguf::Type::Q8_0: return "gemm_glu_q8_0";
    case gguf::Type::Q6_K: return "gemm_glu_q6_k";
    case gguf::Type::Q5_K: return "gemm_glu_q5_k";
    case gguf::Type::Q4_K: return "gemm_glu_q4_k";
    case gguf::Type::IQ4_XS: return "gemm_glu_iq4_xs";
    case gguf::Type::IQ3_XXS: return "gemm_glu_iq3_xxs";
    case gguf::Type::IQ3_S: return "gemm_glu_iq3_s";
    case gguf::Type::IQ2_XXS: return "gemm_glu_iq2_xxs";
    case gguf::Type::IQ2_XS: return "gemm_glu_iq2_xs";
    case gguf::Type::IQ2_S: return "gemm_glu_iq2_s";
    case gguf::Type::IQ1_S: return "gemm_glu_iq1_s";
    case gguf::Type::IQ1_M: return "gemm_glu_iq1_m";
    default: return nullptr;
    }
}

} // namespace

bool Gemm::supports_glu(gguf::Type type, uint32_t k)
{
    return glu_name(type) && k % 256 == 0; // the kernel has no remainder step
}

void Gemm::encode_glu(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                      const MTL::Buffer* wg, uint64_t wg_off, const MTL::Buffer* wu, uint64_t wu_off,
                      const MTL::Buffer* xh, uint64_t xh_off, const MTL::Buffer* y, uint64_t y_off, float limit,
                      uint32_t ldy, uint32_t ldx)
{
    if (!supports_glu(type, k))
        throw std::runtime_error(std::format("no gate-up GEMM for {} with k={}", gguf::traits(type).name, k));
    const auto tr = gguf::traits(type);
    GemmArgs a{t, k, n, uint32_t(uint64_t(k) / tr.block_elems * tr.block_bytes), ldy ? ldy : n, ldx ? ldx : k};
    constexpr uint32_t tn = 16; // outputs a threadgroup
    enc->setBuffer(wg, wg_off, 0);
    enc->setBuffer(wu, wu_off, 4);
    enc->setBytes(&limit, sizeof(limit), 5);
    uint32_t t0 = 0;
    for (const TileRun& run : plan_tiles(type, t)) {
        a.t = std::min(t - t0, run.tm * run.count);
        enc->setComputePipelineState(pso(std::format("{}_{}x32", glu_name(type), run.tm)));
        enc->setBuffer(xh, xh_off + uint64_t(t0) * a.ldx * 2, 1);
        enc->setBuffer(y, y_off + uint64_t(t0) * a.ldy * 2, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + tn - 1) / tn, run.count, 1), MTL::Size(128, 1, 1));
        t0 += a.t;
    }
}

void Gemm::encode(MTL::ComputeCommandEncoder* enc, gguf::Type type, uint32_t k, uint32_t n, uint32_t t,
                  const MTL::Buffer* w, uint64_t w_off, const MTL::Buffer* xh, uint64_t xh_off, const MTL::Buffer* y,
                  uint64_t y_off, uint32_t ldy, uint32_t ldx, bool direct, const Batch& batch)
{
    if (batch.count > 1 && (type == gguf::Type::I8 || type == gguf::Type::BF16 || direct ||
                            (few_name(type) && t <= few_tokens_)))
        throw std::runtime_error("batched GEMM: only the staged kernels");
    const char* name = kernel_name(type);
    if (!name || k % 32)
        throw std::runtime_error(std::format("no GEMM for {} with k={}", gguf::traits(type).name, k));
    const auto tr = gguf::traits(type);
    GemmArgs a{t, k, n, uint32_t(uint64_t(k) / tr.block_elems * tr.block_bytes), ldy ? ldy : n, ldx ? ldx : k};
    if (const char* few = few_name(type); few && t <= few_tokens_) {
        // Weights dequantized once per pass of up to 4 tokens, in registers; 16 rows per threadgroup.
        const uint32_t nt = t <= 1 ? 1 : t <= 2 ? 2 : 4;
        enc->setComputePipelineState(pso(std::format("{}_{}", few, nt)));
        enc->setBuffer(w, w_off, 0);
        enc->setBuffer(xh, xh_off, 1);
        enc->setBuffer(y, y_off, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
        return;
    }
    if (type == gguf::Type::I8) {
        if (k % 128 || (ldx && ldx != k))
            throw std::runtime_error(std::format("int8 GEMM: k={} is not a multiple of 128, or x is strided", k));
        const bool wide = t > 64 && n % 128 == 0;
        const uint32_t bm = t <= 32 ? 32 : t <= 64 || wide ? 64 : 128, bn = wide ? 128 : 64;
        enc->setComputePipelineState(pso(std::format("gemm_i8_{}x{}", bm, bn)));
        enc->setBuffer(w, w_off, 0);
        enc->setBuffer(xh, xh_off, 1);
        enc->setBuffer(y, y_off, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + bn - 1) / bn, (t + bm - 1) / bm, 1), MTL::Size(128, 1, 1));
        return;
    }
    if (type == gguf::Type::BF16 || (direct && type == gguf::Type::F16)) {
        if (k % 64)
            throw std::runtime_error(std::format("dense GEMM: k={} is not a multiple of 64", k));
        // Past 64 tokens, 64 x 128 tiles where n allows: half the passes over the activations.
        const bool wide = t > 64 && n % 128 == 0;
        const uint32_t bm = t <= 32 ? 32 : t <= 64 || wide ? 64 : 128, bn = wide ? 128 : 64;
        enc->setComputePipelineState(
            pso(std::format("{}_{}x{}", type == gguf::Type::BF16 ? "gemm_bf16" : "gemm_f16d", bm, bn)));
        enc->setBuffer(w, w_off, 0);
        enc->setBuffer(xh, xh_off, 1);
        enc->setBuffer(y, y_off, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + bn - 1) / bn, (t + bm - 1) / bm, 1), MTL::Size(128, 1, 1));
        return;
    }
    constexpr uint32_t tn = 32;
    enc->setBuffer(w, w_off, 0);
    if (batch.count > 1) { // one tile size, the matrices' tiles along the grid's y
        const uint32_t tm = staged_tile(type, t), tiles = (t + tm - 1) / tm;
        a.batch = batch.count;
        a.t_tiles = tiles;
        a.w_mat = batch.w_mat;
        a.x_mat = batch.x_mat;
        a.y_mat = batch.y_mat;
        enc->setComputePipelineState(pso(std::format("{}_{}x{}", name, tm, tn)));
        enc->setBuffer(xh, xh_off, 1);
        enc->setBuffer(y, y_off, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + tn - 1) / tn, tiles * batch.count, 1), MTL::Size(128, 1, 1));
        return;
    }
    uint32_t t0 = 0;
    for (const TileRun& run : plan_tiles(type, t)) {
        a.t = std::min(t - t0, run.tm * run.count);
        enc->setComputePipelineState(pso(std::format("{}_{}x{}", name, run.tm, tn)));
        enc->setBuffer(xh, xh_off + uint64_t(t0) * a.ldx * 2, 1);
        enc->setBuffer(y, y_off + uint64_t(t0) * a.ldy * 4, 2);
        enc->setBytes(&a, sizeof(a), 3);
        enc->dispatchThreadgroups(MTL::Size((n + tn - 1) / tn, run.count, 1), MTL::Size(128, 1, 1));
        t0 += a.t;
    }
}

void Gemm::to_bf16(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                   uint32_t cols, uint32_t ldx, const MTL::Buffer* xb, uint64_t xb_off)
{
    const uint32_t a[2] = {cols, ldx ? ldx : cols};
    enc->setComputePipelineState(pso("to_bf16"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(xb, xb_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    enc->dispatchThreads(MTL::Size(cols, rows, 1), MTL::Size(256, 1, 1));
}

void Gemm::quantize_i8(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                       uint32_t cols, uint32_t ldx, const MTL::Buffer* q, uint64_t q_off, bool from_bf16)
{
    if (cols % 4)
        throw std::runtime_error(std::format("quantize_i8: {} columns are not a multiple of 4", cols));
    const uint32_t a[3] = {cols, ldx ? ldx : cols, rows};
    enc->setComputePipelineState(pso(from_bf16 ? "quant_rows_i8_bf16" : "quant_rows_i8_f32"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(q, q_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
}

void Gemm::to_half(MTL::ComputeCommandEncoder* enc, const MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                   uint32_t cols, uint32_t ldx, const MTL::Buffer* xh, uint64_t xh_off)
{
    const uint32_t a[2] = {cols, ldx ? ldx : cols};
    enc->setComputePipelineState(pso("to_half"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(xh, xh_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    enc->dispatchThreads(MTL::Size(cols, rows, 1), MTL::Size(256, 1, 1));
}

} // namespace mako::mtl
