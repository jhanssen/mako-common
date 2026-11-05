#include "models/qwen_vision.hpp"

#include "log/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mako::models {

namespace {

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

uint16_t to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

struct LnArgs {
    uint32_t n;
    float eps;
};

} // namespace

QwenVision::QwenVision(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry)
    : TransformerRuntime(ctx, telemetry)
{
    load(dir);
}

QwenVision::Tensor QwenVision::matrix(std::span<const float> w, uint32_t k, uint32_t n)
{
    if (w.size() != uint64_t(k) * n)
        throw std::runtime_error("vision tower: a matrix of the wrong size");
    Tensor t = reserve(w.size() * 2, gguf::Type::BF16, {k, n});
    auto* o = reinterpret_cast<uint16_t*>(static_cast<char*>(t.buf->contents()) + t.off);
    for (size_t i = 0; i < w.size(); ++i)
        o[i] = to_bf16(w[i]);
    return t;
}

QwenVision::Tensor QwenVision::vector(std::span<const float> v)
{
    Tensor t = reserve(v.size() * 4, gguf::Type::F32, {v.size()});
    std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, v.data(), v.size() * 4);
    return t;
}

void QwenVision::load(const std::string& dir)
{
    const auto f = open_files(dir);
    const std::string v = "model.visual.";
    if (!f.find(v + "patch_embed.proj.weight"))
        throw std::runtime_error(dir + ": no vision tower (model.visual.*)");
    // The patch embedding's [1152][3][2][16][16] is its [1152][1536] matrix over the patch's values in that order.
    patch_w_ = upload(f, v + "patch_embed.proj.weight", true);
    patch_b_ = upload_f32(f, v + "patch_embed.proj.bias");
    if (patch_w_.dims.size() != 5 || patch_w_.dims[4] != kDim)
        throw std::runtime_error("vision tower: the patch embedding is not [1152][3][2][16][16]");
    patch_w_.dims = {kPixels, kDim};
    pos_ = floats(f, v + "pos_embed.weight");
    if (pos_.size() != uint64_t(kGrid) * kGrid * kDim)
        throw std::runtime_error("vision tower: the position grid is not [2304][1152]");
    for (uint32_t i = 0; f.find(std::format("{}blocks.{}.norm1.weight", v, i)); ++i) {
        const std::string p = std::format("{}blocks.{}.", v, i);
        Block B;
        B.ln1_w = upload_f32(f, p + "norm1.weight");
        B.ln1_b = upload_f32(f, p + "norm1.bias");
        B.ln2_w = upload_f32(f, p + "norm2.weight");
        B.ln2_b = upload_f32(f, p + "norm2.bias");
        // The heads padded from 72 to 128 with zero rows (q, k, v each [16][128]) so the attention kernels take them;
        // the output projection reads the padded columns as zeros.
        {
            const auto w = floats(f, p + "attn.qkv.weight"), b = floats(f, p + "attn.qkv.bias");
            if (w.size() != uint64_t(3) * kHeads * kHeadDim * kDim)
                throw std::runtime_error(p + "attn.qkv: unexpected shape");
            // The queries carry sqrt(128 / 72): the attention kernel scales by the padded width's root.
            const float qs = std::sqrt(float(kHeadPad) / float(kHeadDim));
            std::vector<float> wp(uint64_t(3) * kHeads * kHeadPad * kDim, 0.0f), bp(3 * kHeads * kHeadPad, 0.0f);
            for (uint32_t s = 0; s < 3 * kHeads; ++s)
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    const float scale = s < kHeads ? qs : 1.0f;
                    const float* src = &w[(uint64_t(s) * kHeadDim + d) * kDim];
                    float* dst = &wp[(uint64_t(s) * kHeadPad + d) * kDim];
                    for (uint32_t c = 0; c < kDim; ++c)
                        dst[c] = src[c] * scale;
                    bp[s * kHeadPad + d] = b[s * kHeadDim + d] * scale;
                }
            B.qkv = matrix(wp, kDim, 3 * kHeads * kHeadPad);
            B.qkv_b = vector(bp);
        }
        {
            const auto w = floats(f, p + "attn.proj.weight");
            std::vector<float> wp(uint64_t(kDim) * kHeads * kHeadPad, 0.0f);
            for (uint32_t o = 0; o < kDim; ++o)
                for (uint32_t h = 0; h < kHeads; ++h)
                    std::copy_n(&w[(uint64_t(o) * kHeads + h) * kHeadDim], kHeadDim,
                                &wp[(uint64_t(o) * kHeads + h) * kHeadPad]);
            B.proj = matrix(wp, kHeads * kHeadPad, kDim);
            B.proj_b = upload_f32(f, p + "attn.proj.bias");
        }
        // The MLP's 4304 hidden values padded to 4352 (the GEMM's multiple of 64), the padding's outputs zero.
        {
            const auto w1 = floats(f, p + "mlp.linear_fc1.weight"), b1 = floats(f, p + "mlp.linear_fc1.bias");
            const auto w2 = floats(f, p + "mlp.linear_fc2.weight");
            std::vector<float> w1p(uint64_t(kFfnPad) * kDim, 0.0f), b1p(kFfnPad, 0.0f), w2p(uint64_t(kDim) * kFfnPad, 0.0f);
            std::copy(w1.begin(), w1.end(), w1p.begin());
            std::copy(b1.begin(), b1.end(), b1p.begin());
            for (uint32_t o = 0; o < kDim; ++o)
                std::copy_n(&w2[uint64_t(o) * kFfn], kFfn, &w2p[uint64_t(o) * kFfnPad]);
            B.fc1 = matrix(w1p, kDim, kFfnPad);
            B.fc1_b = vector(b1p);
            B.fc2 = matrix(w2p, kFfnPad, kDim);
            B.fc2_b = upload_f32(f, p + "mlp.linear_fc2.bias");
        }
        blocks_.push_back(std::move(B));
    }
    if (blocks_.size() != 27)
        throw std::runtime_error("vision tower: not 27 blocks");
    auto merger = [&](const std::string& p) {
        Merger m;
        m.ln_w = upload_f32(f, p + "norm.weight");
        m.ln_b = upload_f32(f, p + "norm.bias");
        m.fc1 = upload(f, p + "linear_fc1.weight", true);
        m.fc1_b = upload_f32(f, p + "linear_fc1.bias");
        m.fc2 = upload(f, p + "linear_fc2.weight", true);
        m.fc2_b = upload_f32(f, p + "linear_fc2.bias");
        if (m.fc1.dims[0] != kMerged || m.fc2.dims[1] != kOut)
            throw std::runtime_error(p + ": unexpected shapes");
        return m;
    };
    merger_ = merger(v + "merger.");
    for (uint32_t j = 0; j < kDeepstack; ++j)
        deep_[j] = merger(std::format("{}deepstack_merger_list.{}.", v, j));
    log::info("vision tower: {} blocks, {:.2f} GB", blocks_.size(), double(dense_bytes_) / 1e9);
}

void QwenVision::layer_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& w, const Tensor& b,
                            MTL::Buffer* y, uint32_t rows, uint32_t n)
{
    const LnArgs a{n, eps_};
    dispatch_at(enc, "layer_norm", {Arg{x, 0}, Arg{w.buf, w.off}, Arg{b.buf, b.off}, Arg{y, 0}}, &a, sizeof(a),
                MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1), false);
}

void QwenVision::merge(MTL::ComputeCommandEncoder*& enc, const Merger& mg, MTL::Buffer* x, uint32_t m, bool post,
                       MTL::Buffer* t, MTL::Buffer* out)
{
    // The main merger norms each patch; the DeepStack ones the four patches of a token together.
    if (post)
        layer_norm(enc, x, mg.ln_w, mg.ln_b, t, m, kMerged);
    else
        layer_norm(enc, x, mg.ln_w, mg.ln_b, t, m * kMerge * kMerge, kDim);
    MTL::Buffer* ff = scratch("v_mf", uint64_t(m) * kMerged * 4);
    proj(enc, mg.fc1, t, 0, ff, 0, m);
    bias(enc, ff, mg.fc1_b, kMerged, m);
    unary(enc, "gelu_erf", ff, m * kMerged);
    proj(enc, mg.fc2, ff, 0, out, 0, m);
    bias(enc, out, mg.fc2_b, kOut, m);
}

QwenVision::Features QwenVision::encode(std::span<const uint8_t> first, std::span<const uint8_t> second, uint32_t h,
                                        uint32_t w)
{
    const uint32_t gh = h / kPatch, gw = w / kPatch;
    if (!h || !w || h % (kPatch * kMerge) || w % (kPatch * kMerge) || first.size() != uint64_t(h) * w * 3 ||
        second.size() != first.size())
        throw std::runtime_error("vision tower: frames [h][w][3] with h and w multiples of 32");
    const std::span<const uint8_t> taps[2] = {first, second};
    const uint32_t n = gh * gw, m = n / (kMerge * kMerge), bw = gw / kMerge;
    const uint32_t q_pad = round_up(n, kAttnRows), k_pad = round_up(n, 128);
    // The patches in block order (2x2 merge blocks row-major, the patches within a block row-major), each its
    // [3][2][16][16] values in [-1, 1], a frame a time tap, as the processors lay them out; their position
    // embeddings and rope angles from their (row, column) in the patch grid.
    MTL::Buffer* pix = scratch("v_pix", uint64_t(n) * kPixels * 4);
    MTL::Buffer* pos = scratch("v_pos", uint64_t(n) * kDim * 4);
    MTL::Buffer* cs = scratch("v_cs", uint64_t(n) * (kHeadDim / 2) * 8);
    auto* pf = static_cast<float*>(pix->contents());
    auto* pe = static_cast<float*>(pos->contents());
    auto* cf = static_cast<float*>(cs->contents());
    float inv[kHeadDim / 4];
    for (uint32_t i = 0; i < kHeadDim / 4; ++i)
        inv[i] = 1.0f / std::pow(theta_, float(2 * i) / float(kHeadDim / 2));
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t row = (i / (kMerge * kMerge) / bw) * kMerge + (i / kMerge) % kMerge;
        const uint32_t col = (i / (kMerge * kMerge) % bw) * kMerge + i % kMerge;
        float* p = pf + uint64_t(i) * kPixels;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t t = 0; t < 2; ++t)
                for (uint32_t y = 0; y < kPatch; ++y)
                    for (uint32_t x = 0; x < kPatch; ++x)
                        p[((c * 2 + t) * kPatch + y) * kPatch + x] =
                            float(taps[t][((uint64_t(row) * kPatch + y) * w + col * kPatch + x) * 3 + c]) / 127.5f - 1.0f;
        // Bilinear, corners aligned: the grid's 0..47 over the patch rows 0..gh - 1.
        const double sy = gh > 1 ? double(row) * (kGrid - 1) / double(gh - 1) : 0.0;
        const double sx = gw > 1 ? double(col) * (kGrid - 1) / double(gw - 1) : 0.0;
        const uint32_t y0 = std::min(uint32_t(sy), kGrid - 1), x0 = std::min(uint32_t(sx), kGrid - 1);
        const uint32_t y1 = std::min(y0 + 1, kGrid - 1), x1 = std::min(x0 + 1, kGrid - 1);
        const float wy = float(sy - y0), wx = float(sx - x0);
        const float* g00 = &pos_[(uint64_t(y0) * kGrid + x0) * kDim];
        const float* g01 = &pos_[(uint64_t(y0) * kGrid + x1) * kDim];
        const float* g10 = &pos_[(uint64_t(y1) * kGrid + x0) * kDim];
        const float* g11 = &pos_[(uint64_t(y1) * kGrid + x1) * kDim];
        float* o = pe + uint64_t(i) * kDim;
        for (uint32_t d = 0; d < kDim; ++d)
            o[d] = (1 - wy) * ((1 - wx) * g00[d] + wx * g01[d]) + wy * ((1 - wx) * g10[d] + wx * g11[d]);
        // Pairs 0..17 turn by the row, 18..35 by the column; pair j rotates value j with j + 36.
        for (uint32_t j = 0; j < kHeadDim / 2; ++j) {
            const float ang = float(j < kHeadDim / 4 ? row : col) * inv[j % (kHeadDim / 4)];
            cf[(uint64_t(i) * (kHeadDim / 2) + j) * 2] = std::cos(ang);
            cf[(uint64_t(i) * (kHeadDim / 2) + j) * 2 + 1] = std::sin(ang);
        }
    }
    const uint32_t qw = kHeads * kHeadPad;
    MTL::Buffer* x = scratch("v_x", uint64_t(n) * kDim * 4);
    MTL::Buffer* xn = scratch("v_xn", uint64_t(n) * kDim * 4);
    MTL::Buffer* qkv = scratch("v_qkv", uint64_t(n) * 3 * qw * 4);
    MTL::Buffer* qh = scratch("v_qh", uint64_t(kHeads) * q_pad * kHeadPad * 2);
    MTL::Buffer* kh = scratch("v_kh", uint64_t(kHeads) * k_pad * kHeadPad * 2);
    MTL::Buffer* vh = scratch("v_vh", uint64_t(kHeads) * k_pad * kHeadPad * 2);
    MTL::Buffer* att = scratch("v_att", uint64_t(n) * qw * 4);
    MTL::Buffer* y = scratch("v_y", uint64_t(n) * kDim * 4);
    MTL::Buffer* ff = scratch("v_ff", uint64_t(n) * kFfnPad * 4);
    MTL::Buffer* mt = scratch("v_mt", uint64_t(n) * kDim * 4);
    MTL::Buffer* out = scratch("v_out", uint64_t(m) * kOut * 4);
    std::array<MTL::Buffer*, kDeepstack> deep;
    for (uint32_t j = 0; j < kDeepstack; ++j)
        deep[j] = scratch(std::format("v_deep{}", j), uint64_t(m) * kOut * 4);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    // The activations' bfloat rounding carried through the GEMMs as a second pass (16 bits of them): the tower is
    // small, and its 27 blocks compound the rounding otherwise.
    bf16_split_ = true;
    proj(enc, patch_w_, pix, 0, x, 0, n);
    bias(enc, x, patch_b_, kDim, n);
    probe(enc, "patch", -1, x, uint64_t(n) * kDim);
    add(enc, x, pos, x, n * kDim);
    probe(enc, "in", -1, x, uint64_t(n) * kDim);
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        const Block& B = blocks_[i];
        cur_layer_ = i;
        layer_norm(enc, x, B.ln1_w, B.ln1_b, xn, n, kDim);
        proj(enc, B.qkv, xn, 0, qkv, 0, n);
        bias(enc, qkv, B.qkv_b, 3 * qw, n);
        heads_partial(enc, qkv, 3 * qw, 0, kHeadPad, kHeadPad, kHeadDim, kHeads, n, q_pad, cs, nullptr, 0.0f, qh);
        heads_partial(enc, qkv, 3 * qw, qw, kHeadPad, kHeadPad, kHeadDim, kHeads, n, k_pad, cs, nullptr, 0.0f, kh);
        heads_partial(enc, qkv, 3 * qw, 2 * qw, kHeadPad, kHeadPad, 0, kHeads, n, k_pad, nullptr, nullptr, 0.0f, vh);
        if (k_pad > n) {
            const uint32_t c[4] = {kHeadPad, k_pad, n, 0};
            for (MTL::Buffer* b : {kh, vh})
                dispatch(enc, "heads_clear", {b}, c, sizeof(c), MTL::Size(kHeadPad, k_pad - n, kHeads), MTL::Size(128, 1, 1));
        }
        attention(enc, qh, kh, vh, att, n, q_pad, n, k_pad, kHeads, kHeads, 0, 0, kHeadPad);
        proj(enc, B.proj, att, 0, y, 0, n);
        bias(enc, y, B.proj_b, kDim, n);
        add(enc, x, y, x, n * kDim);
        layer_norm(enc, x, B.ln2_w, B.ln2_b, xn, n, kDim);
        proj(enc, B.fc1, xn, 0, ff, 0, n);
        bias(enc, ff, B.fc1_b, kFfnPad, n);
        unary(enc, "gelu_tanh", ff, n * kFfnPad);
        proj(enc, B.fc2, ff, 0, y, 0, n);
        bias(enc, y, B.fc2_b, kDim, n);
        add(enc, x, y, x, n * kDim);
        probe(enc, "block", int(i), x, uint64_t(n) * kDim);
        for (uint32_t j = 0; j < kDeepstack; ++j)
            if (i == kDeepstackBlocks[j])
                merge(enc, deep_[j], x, m, true, mt, deep[j]);
    }
    merge(enc, merger_, x, m, false, mt, out);
    bf16_split_ = false;
    enc->endEncoding();
    finish(cb);
    Features F;
    F.rows = gh / kMerge;
    F.cols = bw;
    const auto* o = static_cast<const float*>(out->contents());
    F.embeds.assign(o, o + uint64_t(m) * kOut);
    for (uint32_t j = 0; j < kDeepstack; ++j) {
        const auto* d = static_cast<const float*>(deep[j]->contents());
        F.deepstack[j].assign(d, d + uint64_t(m) * kOut);
    }
    return F;
}

} // namespace mako::models
