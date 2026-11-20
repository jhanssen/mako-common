#include "models/h3.hpp"

#include "gguf/read_pool.hpp"
#include "log/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>

namespace mako::models {

namespace {

struct RowModArgs {
    uint32_t n;
    float eps;
    uint32_t shift_off, scale_off;
};
struct GateRowsArgs {
    uint32_t n, gate_off;
};

constexpr float kEps = 1e-5f;
constexpr uint32_t kFfn = 14336, kQkv = 3 * H3::kHeads * 128, kAdaln = 6 * 3 * H3::kDim, kModality = 6 * H3::kDim;
constexpr uint32_t kOutRows = 256; // the velocity heads' rows, padded for the GEMM
// Rotary time: a latent frame spans 5/3 of these units for each of its (1, 4, 4, 4, 4) pixel frames; a spatial axis
// is normalized by the square root of the latent area and scaled by 32.
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int kFramesPerLatent[5] = {1, 4, 4, 4, 4};
constexpr double kSpatialScale = 32.0;

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

std::optional<uint32_t> H3::parse_int8(std::string_view s)
{
    if (s == "none" || s.empty())
        return 0u;
    if (s == "all")
        return uint32_t(kInt8All);
    uint32_t mask = 0;
    for (size_t at = 0; at <= s.size();) {
        const size_t comma = std::min(s.find(',', at), s.size());
        const std::string_view part = s.substr(at, comma - at);
        if (part == "qkv")
            mask |= kInt8Qkv;
        else if (part == "out")
            mask |= kInt8Out;
        else if (part == "fc1")
            mask |= kInt8Fc1;
        else if (part == "fc2")
            mask |= kInt8Fc2;
        else
            return std::nullopt;
        at = comma + 1;
    }
    return mask;
}

H3::H3(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry, uint32_t int8, const std::string& heads)
    : TransformerRuntime(ctx, telemetry), int8_(int8 & kInt8All)
{
    const auto f = open_files(path);
    // The shards interleave each head's q, k and v rows; the published single files stack them. Nothing in the file
    // tells them apart.
    interleaved_ = std::filesystem::is_directory(path);
    load(f, heads);
}

H3::Tensor H3::upload_qkv(const safetensors::Files& f, const std::string& name, bool int8)
{
    const auto r = f.find(name);
    if (!r || r.tensor->type != gguf::Type::BF16 || r.tensor->dims.size() != 2 || r.tensor->dims[0] != kDim ||
        r.tensor->dims[1] != kQkv)
        throw std::runtime_error(name + ": not the BF16 [21504][5376] fused q / k / v matrix");
    if (!interleaved_)
        return int8 ? upload_i8(f, name)
                    : upload_raw(r.file->fd(), r.tensor->offset, r.tensor->bytes, gguf::Type::BF16, r.tensor->dims,
                                 name, true);
    // The restacked rows go straight into the resident BF16 copy, or through the staging buffer when quantized.
    Tensor t;
    char* dst;
    if (int8) {
        dst = static_cast<char*>(scratch("i8_stage", r.tensor->bytes)->contents());
    } else {
        t = reserve(r.tensor->bytes, gguf::Type::BF16, r.tensor->dims);
        dst = static_cast<char*>(t.buf->contents()) + t.off;
    }
    const uint64_t row = uint64_t(kDim) * 2, head = 128 * row;
    std::vector<char> tmp(3 * head);
    for (uint32_t h = 0; h < kHeads; ++h) { // head h's [q k v] rows to q, k and v's row blocks
        gguf::read_exact(r.file->fd(), tmp.data(), tmp.size(), r.tensor->offset + uint64_t(h) * 3 * head);
        for (uint32_t which = 0; which < 3; ++which)
            std::memcpy(dst + (uint64_t(which) * kHeads + h) * head, tmp.data() + which * head, head);
    }
    return int8 ? quantize_i8(scratch("i8_stage", r.tensor->bytes), 0, kDim, kQkv, r.tensor->dims) : t;
}

H3::Tensor H3::upload_matrix(const safetensors::Files& f, const std::string& name)
{
    // F16 matrices (the ComfyUI repack's modulation projections) stay F16; the GEMM and the curve kernel read both.
    const auto r = f.find(name);
    if (!r || r.tensor->dims.size() != 2)
        throw std::runtime_error(name + ": missing, or not a matrix");
    if (r.tensor->type != gguf::Type::F16)
        return upload(f, name, true);
    Tensor t = reserve(r.tensor->bytes, gguf::Type::F16, r.tensor->dims);
    gguf::read_exact(r.file->fd(), static_cast<char*>(t.buf->contents()) + t.off, r.tensor->bytes, r.tensor->offset);
    return t;
}

H3::Tensor H3::upload_padded(const safetensors::Files& f, const std::string& name, uint32_t rows)
{
    const auto r = f.find(name);
    if (!r || r.tensor->dims.size() != 2 || r.tensor->dims[1] > rows)
        throw std::runtime_error(name + ": missing, or not a matrix of at most " + std::to_string(rows) + " rows");
    const uint32_t k = uint32_t(r.tensor->dims[0]);
    const auto v = read_floats(*r.file, *r.tensor);
    Tensor t = reserve(uint64_t(rows) * k * 4, gguf::Type::F32, {k, rows});
    auto* o = reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off);
    std::fill(o, o + uint64_t(rows) * k, 0.0f);
    std::copy(v.begin(), v.end(), o);
    return t;
}

void H3::add_head(std::span<const float> w, std::span<const float> b, uint32_t n)
{
    if (n > kOutRows || w.size() != uint64_t(n) * kDim || b.size() != n)
        throw std::runtime_error("transformer: a video head is not [96][5376] with its bias");
    Tensor t = reserve(uint64_t(kOutRows) * kDim * 4, gguf::Type::F32, {kDim, kOutRows});
    auto* o = reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off);
    std::fill(o, o + uint64_t(kOutRows) * kDim, 0.0f);
    std::copy(w.begin(), w.end(), o);
    Tensor tb = reserve(uint64_t(n) * 4, gguf::Type::F32, {n});
    std::copy(b.begin(), b.end(), reinterpret_cast<float*>(static_cast<char*>(tb.buf->contents()) + tb.off));
    video_out_.push_back(t);
    video_out_b_.push_back(tb);
}

uint32_t H3::blend_heads(std::span<const float> weights)
{
    if (weights.size() != video_out_.size())
        throw std::runtime_error(std::format("blend_heads: {} weights for {} heads", weights.size(), video_out_.size()));
    if (!video_out_blend_.buf) {
        video_out_blend_ = reserve(uint64_t(kOutRows) * kDim * 4, gguf::Type::F32, {kDim, kOutRows});
        video_out_blend_b_ = reserve(uint64_t(kVideoPatch) * 4, gguf::Type::F32, {kVideoPatch});
    }
    auto at = [](const Tensor& t) { return reinterpret_cast<float*>(static_cast<char*>(t.buf->contents()) + t.off); };
    float* o = at(video_out_blend_);
    float* ob = at(video_out_blend_b_);
    std::fill(o, o + uint64_t(kOutRows) * kDim, 0.0f);
    std::fill(ob, ob + kVideoPatch, 0.0f);
    for (size_t s = 0; s < weights.size(); ++s) {
        if (weights[s] == 0.0f)
            continue;
        const float* w = at(video_out_[s]);
        const float* b = at(video_out_b_[s]);
        const uint64_t nb = video_out_b_[s].dims[0];
        for (uint64_t i = 0; i < uint64_t(kVideoPatch) * kDim; ++i)
            o[i] += weights[s] * w[i];
        for (uint64_t i = 0; i < std::min<uint64_t>(nb, kVideoPatch); ++i)
            ob[i] += weights[s] * b[i];
    }
    return kBlendHead;
}

void H3::load_heads(const safetensors::Files& f, const std::string& heads)
{
    const uint32_t n = kVideoPatch;
    if (!heads.empty()) {
        const auto hf = open_files(heads);
        const auto r = hf.find("video_out.weight"), rb = hf.find("video_out.bias");
        if (!r || !rb || r.tensor->dims.size() != 3 || r.tensor->dims[0] != kDim || r.tensor->dims[1] != n ||
            rb.tensor->dims.size() != 2 || rb.tensor->dims[1] != r.tensor->dims[2])
            throw std::runtime_error(heads + ": not video_out.weight [steps][96][5376] with its bias");
        const auto w = read_floats(*r.file, *r.tensor), b = read_floats(*rb.file, *rb.tensor);
        const uint32_t steps = uint32_t(r.tensor->dims[2]);
        for (uint32_t s = 0; s < steps; ++s)
            add_head(std::span(w).subspan(uint64_t(s) * n * kDim, uint64_t(n) * kDim), std::span(b).subspan(uint64_t(s) * n, n), n);
        return;
    }
    const auto r = f.find("final_layer.video_out.weight"), rb = f.find("final_layer.video_out.bias");
    if (!r || !rb || r.tensor->dims.size() != 2 || r.tensor->dims[0] != kDim || r.tensor->dims[1] % n ||
        rb.tensor->dims.size() != 1 || rb.tensor->dims[0] != r.tensor->dims[1])
        throw std::runtime_error("final_layer.video_out: not [steps x 96][5376] with its bias");
    const auto w = read_floats(*r.file, *r.tensor), b = read_floats(*rb.file, *rb.tensor);
    const uint32_t steps = uint32_t(r.tensor->dims[1] / n);
    std::vector<float> hw(uint64_t(n) * kDim), hb(n);
    for (uint32_t s = 0; s < steps; ++s) {
        // Head 0 as stored; the others are stored as their difference from it.
        for (uint64_t i = 0; i < uint64_t(n) * kDim; ++i)
            hw[i] = w[i] + (s ? w[uint64_t(s) * n * kDim + i] : 0.0f);
        for (uint32_t i = 0; i < n; ++i)
            hb[i] = b[i] + (s ? b[uint64_t(s) * n + i] : 0.0f);
        add_head(hw, hb, n);
    }
}

H3::Block H3::load_block(const safetensors::Files& f, const std::string& p, bool modulated)
{
    Block B;
    B.norm1 = upload_f32(f, p + "norm1.weight");
    B.norm2 = upload_f32(f, p + "norm2.weight");
    const uint32_t q = modulated ? int8_ : 0; // the refiner's two blocks stay BF16
    auto matrix = [&](const std::string& n, uint32_t flag) { return q & flag ? upload_i8(f, n) : upload(f, n, true); };
    B.qkv = upload_qkv(f, p + "attn.qkv_proj.weight", q & kInt8Qkv);
    B.o = matrix(p + "attn.out_proj.weight", kInt8Out);
    B.norm_q = upload_f32(f, p + "attn.q_norm.weight");
    B.norm_k = upload_f32(f, p + "attn.k_norm.weight");
    B.fc1 = matrix(p + "mlp.fc1.weight", kInt8Fc1);
    B.fc2 = matrix(p + "mlp.fc2.weight", kInt8Fc2);
    if (B.o.dims[0] != kHeads * 128 || B.o.dims[1] != kDim || B.fc1.dims[0] != kDim || B.fc1.dims[1] != 2 * kFfn ||
        B.fc2.dims[0] != kFfn || B.fc2.dims[1] != kDim || B.norm_q.dims[0] != 128)
        throw std::runtime_error(p + ": unexpected shapes");
    if (modulated) {
        B.adaln = upload_matrix(f, p + "adaln_proj.linear.weight");
        B.adaln_b = upload_f32(f, p + "adaln_proj.linear.bias");
        if (B.adaln.dims[0] != (curve_ ? kCurve : kTimeDim) || B.adaln.dims[1] != kAdaln)
            throw std::runtime_error(p + (curve_ ? ": the modulation projection is not [96768][8]"
                                                 : ": the modulation projection is not [96768][2688]"));
    }
    return B;
}

void H3::load(const safetensors::Files& f, const std::string& heads)
{
    curve_ = static_cast<bool>(f.find("adaln_t_table"));
    if (!f.find("blocks.0.attn.qkv_proj.weight"))
        throw std::runtime_error("transformer: not MiniMax-H3's weights");
    for (uint32_t i = 0; f.find(std::format("token_refiner.blocks.{}.attn.qkv_proj.weight", i)); ++i)
        refiner_.push_back(load_block(f, std::format("token_refiner.blocks.{}.", i), false));
    refiner_norm_ = upload_f32(f, "token_refiner.final_norm.weight");
    for (uint32_t i = 0; f.find(std::format("blocks.{}.attn.qkv_proj.weight", i)); ++i)
        blocks_.push_back(load_block(f, std::format("blocks.{}.", i), true));
    if (blocks_.empty() || refiner_.empty())
        throw std::runtime_error("transformer: no blocks");
    cond_ = upload(f, "condition_proj.weight", true);
    cond_b_ = upload_f32(f, "condition_proj.bias");
    // The patch projections, the time embedder and the heads are F32 in the checkpoint and stay so (the GEMM runs
    // them in half).
    video_in_ = upload(f, "video_patch_proj.weight", false);
    video_in_b_ = upload_f32(f, "video_patch_proj.bias");
    const bool audio = static_cast<bool>(f.find("audio_patch_proj.weight"));
    if (audio) {
        audio_in_ = upload(f, "audio_patch_proj.weight", false);
        audio_in_b_ = upload_f32(f, "audio_patch_proj.bias");
    }
    if (curve_) {
        table_ = floats(f, "adaln_t_table");
        if (table_.size() != uint64_t(kCurvePoints) * kCurve)
            throw std::runtime_error("adaln_t_table: not [1025][8]");
    } else {
        t1_ = upload(f, "time_embedder.proj_in.weight", false);
        t1_b_ = upload_f32(f, "time_embedder.proj_in.bias");
        t2_ = upload(f, "time_embedder.proj_out.weight", false);
        t2_b_ = upload_f32(f, "time_embedder.proj_out.bias");
    }
    final_adaln_ = upload_matrix(f, "final_layer.adaln_proj.linear.weight");
    final_adaln_b_ = upload_f32(f, "final_layer.adaln_proj.linear.bias");
    final_norm_ = upload_f32(f, "final_layer.norm.weight");
    load_heads(f, heads);
    if (audio) {
        audio_out_ = upload_padded(f, "final_layer.audio_out.weight", kOutRows);
        audio_out_b_ = upload_f32(f, "final_layer.audio_out.bias");
    }
    if (cond_.dims[0] != kText || cond_.dims[1] != kDim || video_in_.dims[0] != kVideoPatch ||
        (audio && (audio_in_.dims[0] != kAudioChannels || audio_out_b_.dims[0] != kAudioChannels)) ||
        (!curve_ && (t1_.dims[0] != kTimeFreq || t1_.dims[1] != kDim || t2_.dims[0] != kDim ||
                     t2_.dims[1] != kTimeDim)) ||
        final_adaln_.dims[0] != (curve_ ? kCurve : kTimeDim) || final_adaln_.dims[1] != 2 * kDim)
        throw std::runtime_error("transformer: unexpected shapes");
    drop_scratch("i8_stage");
    std::string int8;
    for (const auto& [flag, n] : {std::pair{kInt8Qkv, "qkv"}, std::pair{kInt8Out, "out"}, std::pair{kInt8Fc1, "fc1"},
                                  std::pair{kInt8Fc2, "fc2"}})
        if (int8_ & flag)
            int8 += (int8.empty() ? "" : ", ") + std::string(n);
    log::info("MiniMax-H3 transformer: {} blocks, {} refiner blocks, {:.1f} GB{}{}{}{}", blocks_.size(),
              refiner_.size(), double(dense_bytes_) / 1e9, curve_ ? ", pruned (modulation from the curve table)" : "",
              audio ? "" : ", no audio stream", video_out_.size() > 1 ? std::format(", {} video heads", video_out_.size()) : "",
              int8.empty() ? "" : " (int8: " + int8 + ")");
}

void H3::block(MTL::ComputeCommandEncoder*& enc, const Block& B, MTL::Buffer* x, uint32_t rows, uint32_t Tg,
               MTL::Buffer* mods, uint64_t mods_off, MTL::Buffer* off, MTL::Buffer* cs)
{
    const uint32_t D = kDim, q_pad = round_up(rows, kAttnRows), k_pad = round_up(rows, 128);
    const uint32_t tile_rows = cs && cubes_.n ? round_up(cubes_.rows, 128) : k_pad;
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("d_qkv", uint64_t(Tg) * kQkv * 4);
    MTL::Buffer* qh = scratch("d_qh", uint64_t(kHeads) * tile_rows * 128 * 2);
    MTL::Buffer* kh = scratch("d_kh", uint64_t(kHeads) * tile_rows * 128 * 2);
    MTL::Buffer* vh = scratch("d_vh", uint64_t(kHeads) * tile_rows * 128 * 2);
    MTL::Buffer* att = scratch("d_att", uint64_t(Tg) * kHeads * 128 * 4);
    MTL::Buffer* y = scratch("d_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* gu = scratch("d_gu", uint64_t(Tg) * 2 * kFfn * 4);
    MTL::Buffer* ffm = scratch("d_ffm", uint64_t(Tg) * kFfn * 4);
    // y = rms(x) * w * (1 + scale) + shift per row (the refiner: plain RMS norms).
    auto norm_mod = [&](const Tensor& w, uint32_t shift, uint32_t scale) {
        if (!mods) {
            norm(enc, x, &w, xn, rows, D, D, kEps);
            return;
        }
        const RowModArgs a{D, kEps, shift * D, scale * D};
        enc->setComputePipelineState(pso("rms_mod_rows"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(w.buf, w.off, 1);
        enc->setBuffer(mods, mods_off, 2);
        enc->setBuffer(off, 0, 3);
        enc->setBuffer(xn, 0, 4);
        enc->setBytes(&a, sizeof(a), 5);
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "rms_mod_rows");
    };
    auto gated = [&](uint32_t gate) {
        if (!mods) {
            add(enc, x, y, x, rows * D);
            return;
        }
        const GateRowsArgs a{D, gate * D};
        enc->setComputePipelineState(pso("gate_add_rows"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(y, 0, 1);
        enc->setBuffer(mods, mods_off, 2);
        enc->setBuffer(off, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreads(MTL::Size(D, rows, 1), MTL::Size(256, 1, 1));
        after(enc, "gate_add_rows");
    };
    norm_mod(B.norm1, 0, 1);
    proj(enc, B.qkv, xn, 0, qkv, 0, Tg);
    // Heads of 128 with their RMS norms, the first 96 values rotated (the refiner: no rotation). With sparse
    // attention the tiles take the cubes' row order through the map, and the output comes back in the model's.
    const bool sparse = cs && cubes_.n;
    const uint32_t arows = sparse ? cubes_.rows : rows;
    const uint32_t aq_pad = round_up(arows, kAttnRows), ak_pad = round_up(arows, 128);
    MTL::Buffer* map = sparse ? cubes_.map : nullptr;
    heads_partial(enc, qkv, kQkv, 0, 128, 128, cs ? kRot : 0, kHeads, arows, aq_pad, cs, &B.norm_q, kEps, qh, map);
    heads_partial(enc, qkv, kQkv, kHeads * 128, 128, 128, cs ? kRot : 0, kHeads, arows, ak_pad, cs, &B.norm_k, kEps,
                  kh, map);
    heads_partial(enc, qkv, kQkv, 2 * kHeads * 128, 128, 128, 0, kHeads, arows, ak_pad, cs, nullptr, 0.0f, vh, map);
    if (ak_pad > arows) {
        const uint32_t c[4] = {128, ak_pad, arows, 0};
        for (MTL::Buffer* b : {kh, vh})
            dispatch(enc, "heads_clear", {b}, c, sizeof(c), MTL::Size(128, ak_pad - arows, kHeads), MTL::Size(128, 1, 1));
    }
    if (sparse) {
        MTL::Buffer *sel = nullptr, *nsel = nullptr;
        cube_select(enc, qh, kh, kHeads, aq_pad, ak_pad, cubes_.n, cubes_.cnt, cubes_.topk, sel, nsel, sparse_local_,
                    cubes_.stride);
        attention_sparse(enc, qh, kh, vh, att, arows, aq_pad, arows, ak_pad, kHeads, kHeads, kHeads * 128, cubes_.n,
                         cubes_.topk, cubes_.dense0, cubes_.dense1, sel, nsel, cubes_.valid, map);
    } else {
        attention(enc, qh, kh, vh, att, rows, q_pad, rows, k_pad, kHeads, kHeads, 0, 0);
    }
    proj(enc, B.o, att, 0, y, 0, Tg);
    gated(2);
    norm_mod(B.norm2, 3, 4);
    proj(enc, B.fc1, xn, 0, gu, 0, Tg);
    uint32_t nff = kFfn;
    dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, Tg, 1), MTL::Size(256, 1, 1));
    proj(enc, B.fc2, ffm, 0, y, 0, Tg);
    gated(5);
}

H3::Rows H3::rows(const Layout& l)
{
    Rows R;
    const uint32_t T = l.text, S = l.rows();
    R.pos.assign(uint64_t(S) * 3, 0.0);
    R.tags.assign(S, kVideoTag);
    R.roles.assign(S, kRoleVideo);
    for (uint32_t i = 0; i < T; ++i) {
        R.pos[uint64_t(i) * 3] = i;
        R.tags[i] = l.text_tags.empty() ? kTextTag : l.text_tags[i];
        R.roles[i] = kRoleText;
    }
    // One spatial axis of a block's grid (latent h x w): its n coordinates centred on the unit interval in the
    // block's aspect ratio, times 32 (numpy's linspace without the endpoint).
    auto grid = [](uint32_t h, uint32_t w, uint32_t dim, uint32_t n, uint32_t i) {
        const double ratio = double(dim) / std::sqrt(double(h) * double(w)), left = (1.0 - ratio) / 2.0;
        return (left + double(i) * (ratio / double(n))) * kSpatialScale;
    };
    auto span_of = [](uint32_t f) { return kFrameRescale * kFramesPerLatent[f % 5]; };
    uint32_t cur = T, vsrc = 0, asrc = 0; // the next sequence row, and the next video / audio row step() takes
    double clock = double(T);
    // A block of video rows: frames x grid, from time t0, each frame its span later when `advance`.
    auto video_block = [&](uint32_t frames, uint32_t h, uint32_t w, double t0, bool advance, uint8_t role) {
        const uint32_t ph = h / 2, pw = w / 2;
        R.segments.push_back({false, cur, frames * ph * pw, vsrc});
        double t = t0;
        for (uint32_t f = 0; f < frames; ++f) {
            for (uint32_t y = 0; y < ph; ++y)
                for (uint32_t x = 0; x < pw; ++x) {
                    double* p = &R.pos[uint64_t(cur) * 3];
                    p[0] = t;
                    p[1] = grid(h, w, h, ph, y);
                    p[2] = grid(h, w, w, pw, x);
                    R.roles[cur++] = role;
                    ++vsrc;
                }
            if (advance)
                t += span_of(f);
        }
    };
    // A block of audio rows, channel-major: a latent a time unit from t0, no height, the width axis at the two
    // extremes of the width grid of the block it belongs to (its video's, or the generated video's).
    auto audio_block = [&](uint32_t latents, uint32_t h, uint32_t w, double t0, uint8_t role) {
        if (!latents)
            return;
        R.segments.push_back({true, cur, 2 * latents, asrc});
        const uint32_t pw = w / 2;
        for (uint32_t ch = 0; ch < 2; ++ch)
            for (uint32_t j = 0; j < latents; ++j) {
                double* p = &R.pos[uint64_t(cur) * 3];
                p[0] = t0 + double(j);
                p[2] = grid(h, w, w, pw, ch ? pw - 1 : 0);
                R.tags[cur] = kAudioTag;
                R.roles[cur++] = role;
                ++asrc;
            }
    };
    // Keyframes: at the first generated frame's time, or the last's.
    double total = 0;
    for (uint32_t f = 0; f < l.frames; ++f)
        total += span_of(f);
    for (uint32_t c = 0; c < l.cond_frames; ++c)
        video_block(1, l.h, l.w, l.cond_last[c] ? clock + total - kFrameRescale : clock, false, kRoleCondVideo);
    for (const Layout::Reference& r : l.refs) {
        switch (r.kind) {
        case Layout::Reference::Kind::image: // a single frame takes one integer slot, not a latent frame's 5/3
            video_block(1, r.h, r.w, clock, false, kRoleCondVideo);
            clock += 1.0;
            break;
        case Layout::Reference::Kind::audio:
            audio_block(r.audio, l.h, l.w, clock, kRoleCondAudio);
            clock += double(r.audio);
            break;
        case Layout::Reference::Kind::video: { // the soundtrack's rows right before the video's, sharing its origin
            if (r.audio)
                audio_block(r.audio, r.h, r.w, clock, kRoleCondAudio);
            video_block(r.frames, r.h, r.w, clock, true, kRoleCondVideo);
            double span = 0;
            for (uint32_t f = 0; f < r.frames; ++f)
                span += span_of(f);
            clock += std::max(double(r.audio), span);
            break;
        }
        }
    }
    audio_block(l.audio, l.h, l.w, clock, kRoleAudio);
    video_block(l.frames, l.h, l.w, clock, true, kRoleVideo);
    if (cur != S || vsrc != l.video_rows() || asrc != l.audio_rows())
        throw std::logic_error("layout rows");
    return R;
}

void H3::begin(std::span<const float> text, const Layout& l)
{
    if (!l.text || text.size() != uint64_t(l.text) * kText)
        throw std::runtime_error("text embeddings must be rows of 5120");
    if (!l.frames || !l.h || !l.w || l.h % 2 || l.w % 2)
        throw std::runtime_error("layout: frames and an even latent height and width");
    if (has_audio() ? !l.audio : l.audio || l.cond_audio_rows())
        throw std::runtime_error(has_audio() ? "layout: audio latents" : "layout: this transformer has no audio stream");
    if (!l.text_tags.empty() && l.text_tags.size() != l.text)
        throw std::runtime_error("layout: a tag per text row");
    if (l.cond_last.size() != l.cond_frames)
        throw std::runtime_error("layout: an anchor per conditioning frame");
    if (l.cond_frames && !l.refs.empty())
        throw std::runtime_error("layout: keyframes and references do not go together");
    for (const Layout::Reference& r : l.refs) {
        const bool visual = r.kind != Layout::Reference::Kind::audio;
        if (visual ? (!r.frames || !r.h || !r.w || r.h % 2 || r.w % 2) : !r.audio)
            throw std::runtime_error("layout: a reference needs latent frames and an even grid, or audio latents");
    }
    layout_ = l;
    cond_rows_ = l.cond_rows();
    S_ = l.rows();
    const uint32_t T = l.text, S = S_, Tg = round_up(T, 256);
    MTL::Buffer* tx = scratch("k_tx", uint64_t(T) * kText * 4);
    MTL::Buffer* txt = scratch("k_txt", uint64_t(Tg) * kDim * 4);
    std::memcpy(tx->contents(), text.data(), text.size() * 4);
    std::memset(txt->contents(), 0, txt->length());
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, cond_, tx, 0, txt, 0, Tg > T ? Tg : T);
    bias(enc, txt, cond_b_, kDim, T);
    for (const Block& B : refiner_)
        block(enc, B, txt, T, Tg, nullptr, 0, nullptr, nullptr);
    norm(enc, txt, &refiner_norm_, txt, T, kDim, kDim, kEps);
    probe(enc, "txt", -1, txt, uint64_t(T) * kDim);
    enc->endEncoding();
    finish(cb);

    rows_ = rows(l);
    tags_.assign(rows_.tags.begin(), rows_.tags.end());
    const uint32_t A = l.audio_rows(), rpf = l.rows_per_frame(), ph = l.h / 2, pw = l.w / 2;
    // The sparse attention's cubes: the generated video rows in 4 x 4 x 4 cubes of (frame, patch row, patch column),
    // the grid padded to multiples of 4 with rows that are no token, then the text, conditioning and audio rows.
    cubes_ = {};
    if (sparse_ > 0.0f) {
        const uint32_t fc = (l.frames + kCube - 1) / kCube, yc = (ph + kCube - 1) / kCube, xc = (pw + kCube - 1) / kCube;
        const uint32_t n = fc * yc * xc, vrows = n * 64, dense = T + cond_rows_ + A;
        if (n > 4096)
            throw std::runtime_error("sparse attention: more than 4096 cubes");
        cubes_.n = n;
        cubes_.rows = vrows + dense;
        cubes_.topk = std::max<uint32_t>(1, uint32_t(std::ceil(double(sparse_) * n)));
        cubes_.dense0 = vrows;
        cubes_.dense1 = vrows + dense;
        const uint32_t q_pad = round_up(cubes_.rows, kAttnRows), k_pad = round_up(cubes_.rows, 128);
        cubes_.stride = yc * xc;
        cubes_.map = scratch("k_map", uint64_t(q_pad) * 4);
        cubes_.valid = scratch("k_valid", k_pad);
        cubes_.cnt = scratch("k_cnt", uint64_t(n) * 4);
        auto* map = static_cast<int32_t*>(cubes_.map->contents());
        auto* valid = static_cast<uint8_t*>(cubes_.valid->contents());
        auto* cnt = static_cast<uint32_t*>(cubes_.cnt->contents());
        std::fill_n(map, q_pad, -1);
        std::fill_n(valid, k_pad, uint8_t(0));
        std::fill_n(cnt, n, 0u);
        for (uint32_t f = 0; f < l.frames; ++f)
            for (uint32_t y = 0; y < ph; ++y)
                for (uint32_t x = 0; x < pw; ++x) {
                    const uint32_t cube = ((f / kCube) * yc + y / kCube) * xc + x / kCube;
                    const uint32_t r = cube * 64 + ((f % kCube) * kCube + y % kCube) * kCube + x % kCube;
                    map[r] = int32_t(T + cond_rows_ + A + f * rpf + y * pw + x);
                    valid[r] = 1;
                    ++cnt[cube];
                }
        for (uint32_t r = 0; r < dense; ++r) {
            map[vrows + r] = int32_t(r);
            valid[vrows + r] = 1;
        }
    }
    MTL::Buffer* cs = scratch("k_cs", uint64_t(S) * (kRot / 2) * 8);
    auto* c = static_cast<float*>(cs->contents());
    float inv[kRopeFreqs];
    for (uint32_t j = 0; j < kRopeFreqs; ++j)
        inv[j] = 1.0f / std::pow(kRopeTheta, float(2 * j) / float(2 * kRopeFreqs));
    for (uint32_t r = 0; r < S; ++r)
        for (uint32_t axis = 0; axis < 3; ++axis)
            for (uint32_t j = 0; j < kRopeFreqs; ++j) {
                const float ang = float(rows_.pos[uint64_t(r) * 3 + axis]) * inv[j];
                const uint64_t o = (uint64_t(r) * (kRot / 2) + axis * kRopeFreqs + j) * 2;
                c[o] = std::cos(ang);
                c[o + 1] = std::sin(ang);
            }
}

H3::Velocity H3::step(std::span<const float> video, std::span<const float> audio, float t_video, float t_audio,
                      float t_cond, uint32_t head)
{
    const Layout& l = layout_;
    const uint32_t T = l.text, S = S_, A = l.audio_rows(), V = l.video_rows(), C = cond_rows_, D = kDim;
    if (!S)
        throw std::runtime_error("begin() first");
    if (video.size() != uint64_t(V) * kVideoPatch || audio.size() != uint64_t(A) * kAudioChannels)
        throw std::runtime_error("video rows of 96 and audio rows of 32 as the layout has them");
    if (head == kBlendHead ? !video_out_blend_.buf : head >= video_out_.size())
        throw std::runtime_error(std::format("head {} of {}", head == kBlendHead ? "blend" : std::to_string(head),
                                             video_out_.size()));
    const Tensor& head_w = head == kBlendHead ? video_out_blend_ : video_out_[head];
    const Tensor& head_b = head == kBlendHead ? video_out_blend_b_ : video_out_b_[head];
    const uint32_t Tg = round_up(S, 256), nb = uint32_t(blocks_.size());
    // The distinct timesteps, sorted, and each row's (timestep, tag) modulation: row offsets into a block's [U][96768]
    // modulation output, and into the final layer's [U][10752].
    std::vector<float> ts = {t_video};
    if (has_audio())
        ts.push_back(t_audio);
    if (C)
        ts.push_back(t_cond);
    if (l.cond_audio_rows())
        ts.push_back(kCondAudioT);
    std::ranges::sort(ts);
    ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
    const uint32_t U = uint32_t(ts.size());
    auto index = [&](float t) { return uint32_t(std::ranges::find(ts, t) - ts.begin()); };
    std::vector<uint32_t> off(S), off2(S);
    for (uint32_t r = 0; r < S; ++r) {
        const uint32_t tag = tags_[r];
        const uint8_t role = rows_.roles[r];
        const float t = role == kRoleCondVideo ? t_cond : role == kRoleCondAudio ? kCondAudioT
                        : role == kRoleAudio ? t_audio : t_video;
        const uint32_t u = index(t);
        off[r] = u * kAdaln + tag * kModality;
        off2[r] = u * 2 * D;
    }
    MTL::Buffer* offb = scratch("d_off", uint64_t(S) * 4);
    MTL::Buffer* off2b = scratch("d_off2", uint64_t(S) * 4);
    std::memcpy(offb->contents(), off.data(), off.size() * 4);
    std::memcpy(off2b->contents(), off2.data(), off2.size() * 4);
    // The sinusoidal embedding of each timestep (cos half first), unscaled; or, pruned, its 8 curve coordinates
    // interpolated from the table over [0, 1].
    MTL::Buffer* tp = scratch("d_tp", uint64_t(U) * kTimeFreq * 4);
    auto* tpf = static_cast<float*>(tp->contents());
    for (uint32_t u = 0; u < U; ++u) {
        if (curve_) {
            const float pos = std::clamp(ts[u], 0.0f, 1.0f) * float(kCurvePoints - 1);
            const uint32_t i0 = std::min(uint32_t(pos), uint32_t(kCurvePoints - 2));
            const float w = pos - float(i0);
            for (uint32_t j = 0; j < kCurve; ++j)
                tpf[u * kCurve + j] = (1.0f - w) * table_[i0 * kCurve + j] + w * table_[(i0 + 1) * kCurve + j];
            continue;
        }
        for (uint32_t i = 0; i < kTimeFreq / 2; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * float(i) / float(kTimeFreq / 2));
            const float arg = ts[u] * freq;
            tpf[u * kTimeFreq + i] = std::cos(arg);
            tpf[u * kTimeFreq + kTimeFreq / 2 + i] = std::sin(arg);
        }
    }
    MTL::Buffer* th = scratch("d_th", uint64_t(U) * D * 4);
    MTL::Buffer* temb = scratch("d_temb", uint64_t(U) * kTimeDim * 4);
    MTL::Buffer* mods = scratch("d_mods", uint64_t(nb) * U * kAdaln * 4);
    MTL::Buffer* fmods = scratch("d_fmods", uint64_t(U) * 2 * D * 4);
    MTL::Buffer* x = scratch("d_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("d_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* vin = scratch("d_vin", uint64_t(V) * kVideoPatch * 4);
    MTL::Buffer* ain = scratch("d_ain", std::max<uint64_t>(uint64_t(A) * kAudioChannels * 4, 64));
    MTL::Buffer* vout = scratch("d_vout", uint64_t(V - C) * kOutRows * 4);
    MTL::Buffer* aout = scratch("d_aout", std::max<uint64_t>(uint64_t(A) * kOutRows * 4, 64));
    MTL::Buffer* txt = scratch("k_txt", uint64_t(round_up(T, 256)) * D * 4);
    MTL::Buffer* cs = scratch("k_cs", uint64_t(S) * (kRot / 2) * 8);
    std::memcpy(vin->contents(), video.data(), video.size() * 4);
    if (!audio.empty())
        std::memcpy(ain->contents(), audio.data(), audio.size() * 4);
    std::memset(x->contents(), 0, x->length());
    std::memcpy(x->contents(), txt->contents(), uint64_t(T) * D * 4);

    // At the layers and kernels telemetry levels, the step's GPU time per kernel.
    const bool timing = tel_ >= gpu::Telemetry::layers;
    MTL::Timestamp cpu0 = 0, gpu0 = 0;
    if (timing) {
        add_samples(32768);
        ctx_.device()->sampleTimestamps(&cpu0, &gpu0);
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // The time embedding of each distinct timestep, and silu of it into every block's modulation projection (its
    // input rounded to BF16 as the projection takes it) and the final layer's; pruned, the 8 coordinates straight
    // into the 8-wide projections, in float.
    if (curve_) {
        auto curve = [&](const Tensor& w, const Tensor& b, MTL::Buffer* out, uint64_t out_off, uint32_t n) {
            const uint32_t a[2] = {n, U};
            dispatch_at(enc, w.type == gguf::Type::F16 ? "curve_mods_f16" : "curve_mods",
                        {Arg{w.buf, w.off}, Arg{b.buf, b.off}, Arg{tp, 0}, Arg{out, out_off}},
                        a, sizeof(a), MTL::Size(n, U, 1), MTL::Size(256, 1, 1));
        };
        for (uint32_t b = 0; b < nb; ++b)
            curve(blocks_[b].adaln, blocks_[b].adaln_b, mods, uint64_t(b) * U * kAdaln * 4, kAdaln);
        curve(final_adaln_, final_adaln_b_, fmods, 0, 2 * D);
    } else {
        proj(enc, t1_, tp, 0, th, 0, U);
        bias(enc, th, t1_b_, D, U);
        unary(enc, "silu", th, U * D);
        proj(enc, t2_, th, 0, temb, 0, U);
        bias(enc, temb, t2_b_, kTimeDim, U);
        probe(enc, "temb", -1, temb, uint64_t(U) * kTimeDim);
        unary(enc, "silu", temb, U * kTimeDim);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint64_t o = uint64_t(b) * U * kAdaln * 4;
            proj(enc, blocks_[b].adaln, temb, 0, mods, o, U);
            bias(enc, mods, blocks_[b].adaln_b, kAdaln, U, o);
        }
        proj(enc, final_adaln_, temb, 0, fmods, 0, U);
        bias(enc, fmods, final_adaln_b_, 2 * D, U);
    }
    probe(enc, "mod0", -1, mods, uint64_t(U) * kAdaln);
    // The media rows through their patch projections, segment by segment of the sequence.
    for (const Segment& sg : rows_.segments) {
        const Tensor& w = sg.audio ? audio_in_ : video_in_;
        proj(enc, w, sg.audio ? ain : vin, uint64_t(sg.src) * (sg.audio ? kAudioChannels : kVideoPatch) * 4, x,
             uint64_t(sg.row0) * D * 4, sg.n);
        bias(enc, x, sg.audio ? audio_in_b_ : video_in_b_, D, sg.n, uint64_t(sg.row0) * D * 4);
    }
    probe(enc, "x0", -1, x, uint64_t(S) * D);
    for (uint32_t b = 0; b < nb; ++b) {
        cur_layer_ = b;
        block(enc, blocks_[b], x, S, Tg, mods, uint64_t(b) * U * kAdaln * 4, offb, cs);
        probe(enc, "block", int(b), x, uint64_t(S) * D);
    }
    {
        const RowModArgs a{D, kEps, 0, D};
        enc->setComputePipelineState(pso("rms_mod_rows"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(final_norm_.buf, final_norm_.off, 1);
        enc->setBuffer(fmods, 0, 2);
        enc->setBuffer(off2b, 0, 3);
        enc->setBuffer(xn, 0, 4);
        enc->setBytes(&a, sizeof(a), 5);
        enc->dispatchThreadgroups(MTL::Size(S, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "rms_mod_rows");
    }
    probe(enc, "norm_out", -1, xn, uint64_t(S) * D);
    // The heads over the generated rows: the audio's then the video's, the sequence's last.
    const uint32_t GV = V - C, GA = 2 * l.audio, gv0 = S - GV, ga0 = gv0 - GA;
    proj(enc, head_w, xn, uint64_t(gv0) * D * 4, vout, 0, GV);
    if (GA)
        proj(enc, audio_out_, xn, uint64_t(ga0) * D * 4, aout, 0, GA);
    enc->endEncoding();
    finish(cb);
    if (timing) {
        std::vector<gpu::KernelTime> parts, kernels;
        resolve_times(cpu0, gpu0, [](uint32_t, int, double) {}, parts, kernels);
        std::ranges::sort(kernels, [](const gpu::KernelTime& a, const gpu::KernelTime& b) { return a.ms > b.ms; });
        double total = 0;
        for (const auto& k : kernels)
            total += k.ms;
        log::info("step of {} rows: {:.0f} ms of GPU time in {} kernels", S, total, kernels.size());
        for (size_t i = 0; i < std::min<size_t>(kernels.size(), 16); ++i)
            log::info("  {:7.0f} ms  {:5} x  {}", kernels[i].ms, kernels[i].count, kernels[i].kernel);
    }

    // The heads' rows are padded to 256: the first 96 / 32 of each, plus the bias.
    Velocity v;
    v.video.assign(uint64_t(V) * kVideoPatch, 0.0f);
    v.audio.assign(uint64_t(A) * kAudioChannels, 0.0f);
    const auto* vo = static_cast<const float*>(vout->contents());
    const Tensor& hb = head_b;
    const auto* vb = reinterpret_cast<const float*>(static_cast<const char*>(hb.buf->contents()) + hb.off);
    for (uint32_t r = 0; r < V - C; ++r)
        for (uint32_t i = 0; i < kVideoPatch; ++i)
            v.video[uint64_t(C + r) * kVideoPatch + i] = vo[uint64_t(r) * kOutRows + i] + vb[i];
    if (GA) {
        const auto* ao = static_cast<const float*>(aout->contents());
        const auto* ab = reinterpret_cast<const float*>(static_cast<const char*>(audio_out_b_.buf->contents()) +
                                                        audio_out_b_.off);
        for (uint32_t r = 0; r < GA; ++r)
            for (uint32_t i = 0; i < kAudioChannels; ++i)
                v.audio[uint64_t(A - GA + r) * kAudioChannels + i] = ao[uint64_t(r) * kOutRows + i] + ab[i];
    }
    return v;
}

H3::Tensor* H3::lora_bias(const std::string&) { return nullptr; }

H3::LoraTarget H3::lora_target(const std::string& module)
{
    // Diffusers' names (transformer_blocks.N.attn.to_q, ff.net.0.proj, ...) or the checkpoint's own.
    static constexpr std::pair<const char*, const char*> kParts[] = {
        {".attn.to_q", ".attn.q"},       {".attn.to_k", ".attn.k"},       {".attn.to_v", ".attn.v"},
        {".attn.to_out.0", ".attn.out_proj"}, {".ff.net.0.proj", ".mlp.fc1"}, {".ff.net.2", ".mlp.fc2"},
        {".adaln_proj.linear", ".adaln"}};
    std::string m = module;
    for (const char* p : {"transformer_blocks.", "token_refiner.refiner_blocks."})
        if (m.starts_with(p)) {
            m = (std::string_view(p) == "transformer_blocks." ? "blocks." : "token_refiner.blocks.") +
                m.substr(std::strlen(p));
            break;
        }
    for (const auto& [d, o] : kParts)
        if (const size_t at = m.find(d); at != std::string::npos)
            m.replace(at, std::strlen(d), o);
    Block* B = nullptr;
    std::string_view rest;
    unsigned i = 0;
    int used = 0;
    if (std::sscanf(m.c_str(), "blocks.%u.%n", &i, &used) == 1 && used > 0 && i < blocks_.size()) {
        B = &blocks_[i];
        rest = std::string_view(m).substr(size_t(used));
    } else if (std::sscanf(m.c_str(), "token_refiner.blocks.%u.%n", &i, &used) == 1 && used > 0 &&
               i < refiner_.size()) {
        B = &refiner_[i];
        rest = std::string_view(m).substr(size_t(used));
    }
    if (!B)
        return {};
    const uint32_t hd = kHeads * 128;
    if (rest == "attn.q")
        return {&B->qkv, 0, hd};
    if (rest == "attn.k")
        return {&B->qkv, hd, hd};
    if (rest == "attn.v")
        return {&B->qkv, 2 * hd, hd};
    if (rest == "attn.out_proj")
        return {&B->o, 0, kDim};
    if (rest == "mlp.fc1") // the adapter's rows [value; gate] onto the weights' [gate; value]
        return {&B->fc1, 0, 2 * kFfn, kFfn};
    if (rest == "mlp.fc2")
        return {&B->fc2, 0, kDim};
    if (rest == "adaln" && B->adaln.buf) // on a pruned checkpoint a full model's 2688-wide adapter is skipped
        return {&B->adaln, 0, kAdaln, 0, curve_};
    return {};
}

} // namespace mako::models
