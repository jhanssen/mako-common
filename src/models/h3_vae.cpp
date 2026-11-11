#include "models/h3_vae.hpp"

#include "log/log.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <numbers>
#include <random>
#include <stdexcept>

namespace mako::models {

namespace {

struct LnArgs {
    uint32_t n;
    float eps;
};

constexpr float kEps = 1e-5f, kRopeTheta = 100.0f;
constexpr uint32_t kEmbedCols = 32; // the 24 latent channels padded for the GEMM

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

H3Vae::H3Vae(const std::string& weights, const std::string& config, mtl::Context& ctx, gpu::Telemetry telemetry,
             const std::string& name)
    : ConvRuntime(ctx, telemetry), name_(name), weights_(weights)
{
    {
        std::ifstream in(config, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (!in || glz::read_json(j, text) || !j.contains("latents_mean") || !j.contains("latents_std"))
            throw std::runtime_error(config + ": no latents_mean / latents_std");
        for (const auto& v : j["latents_mean"].get_array())
            mean_.push_back(float(v.get_number()));
        for (const auto& v : j["latents_std"].get_array())
            std_.push_back(float(v.get_number()));
        if (mean_.size() != kLatent || std_.size() != kLatent)
            throw std::runtime_error(config + ": latent statistics are not over 24 channels");
    }
    const auto f = open_files(weights_);
    if (!f.find("decoder.x_embedder.weight") || !f.find("encoder.conv_in.weight"))
        throw std::runtime_error(weights_ + ": not MiniMax-H3's VAE");
    pq_w_ = floats(f, "post_quant_conv.weight"); // [24][24][1][1](1)
    pq_b_ = floats(f, "post_quant_conv.bias");
    if (pq_w_.size() != kLatent * kLatent || pq_b_.size() != kLatent)
        throw std::runtime_error(name_ + ": the post-quant convolution is not 24 -> 24");
    // The voxel embedding, [dim][24], with its input padded to 32 columns.
    {
        const auto w = floats(f, "decoder.x_embedder.weight");
        if (w.size() % kLatent || w.size() / kLatent % kHeadDim)
            throw std::runtime_error(name_ + ": the embedding is not 24 -> a multiple of 64");
        dim_ = uint32_t(w.size() / kLatent);
        embed_ = reserve(uint64_t(dim_) * kEmbedCols * 2, gguf::Type::F16, {kEmbedCols, dim_});
        auto* o = reinterpret_cast<__fp16*>(static_cast<char*>(embed_.buf->contents()) + embed_.off);
        for (uint32_t r = 0; r < dim_; ++r)
            for (uint32_t c = 0; c < kEmbedCols; ++c)
                o[uint64_t(r) * kEmbedCols + c] = __fp16(c < kLatent ? w[uint64_t(r) * kLatent + c] : 0.0f);
    }
    const uint32_t D = dim_;
    embed_b_ = upload_f32(f, "decoder.x_embedder.bias");
    registers_ = upload_f32(f, "decoder.register_tokens");
    if (registers_.dims.empty() || registers_.dims[0] != uint64_t(kRegisters) * D)
        throw std::runtime_error(std::format("{}: not {} register tokens of {}", name_, kRegisters, D));
    ones_ = reserve(kHeadDim * 4, gguf::Type::F32, {kHeadDim});
    std::fill_n(reinterpret_cast<float*>(static_cast<char*>(ones_.buf->contents()) + ones_.off), kHeadDim, 1.0f);
    for (uint32_t i = 0; f.find(std::format("decoder.transformer_blocks.{}.norm1.weight", i)); ++i) {
        const std::string p = std::format("decoder.transformer_blocks.{}.", i);
        Block B;
        B.norm1 = upload_f32(f, p + "norm1.weight");
        B.norm2 = upload_f32(f, p + "norm2.weight");
        B.qkv = upload_f16(f, p + "attn.to_qkv.weight");
        B.qkv_b = upload_f32(f, p + "attn.to_qkv.bias");
        B.o = upload_f16(f, p + "attn.to_out.weight");
        B.o_b = upload_f32(f, p + "attn.to_out.bias");
        B.w1 = upload_f16(f, p + "ff.w1.weight");
        B.w1_b = upload_f32(f, p + "ff.w1.bias");
        B.w2 = upload_f16(f, p + "ff.w2.weight");
        B.w2_b = upload_f32(f, p + "ff.w2.bias");
        B.scale1 = upload_f32(f, p + "scale1");
        B.scale2 = upload_f32(f, p + "scale2");
        if (i == 0)
            ffn_ = uint32_t(B.w1.dims[1] / 2);
        if (B.qkv.dims[0] != D || B.qkv.dims[1] != 3 * D || B.o.dims[0] != D || B.o.dims[1] != D ||
            B.w1.dims[1] != 2 * ffn_ || B.w2.dims[0] != ffn_ || B.w2.dims[1] != D)
            throw std::runtime_error(p + ": unexpected shapes");
        blocks_.push_back(std::move(B));
    }
    if (blocks_.empty())
        throw std::runtime_error(name_ + ": no decoder blocks");
    out_norm_w_ = upload_f32(f, "decoder.norm_out.weight");
    out_norm_b_ = upload_f32(f, "decoder.norm_out.bias");
    proj_out_ = upload_f16(f, "decoder.proj_out.weight");
    proj_out_b_ = upload_f32(f, "decoder.proj_out.bias");
    // The patch projection: [dim][3 x frames x 16 x 16].
    if (proj_out_.dims[0] != D || proj_out_.dims[1] % (3 * kPatch * kPatch) || !proj_out_.dims[1])
        throw std::runtime_error(name_ + ": the patch projection is not to 16x16 pixel patches");
    patch_t_ = uint32_t(proj_out_.dims[1] / (3 * kPatch * kPatch));
    log::info("{} decoder: {} blocks of {}, a token {} frame{} deep, {:.2f} GB", name_, blocks_.size(), D, patch_t_,
              patch_t_ == 1 ? "" : "s", double(dense_bytes_) / 1e9);
}

std::vector<H3Vae::Tile> H3Vae::tiles(uint32_t length)
{
    // The fewest 256-pixel tiles that cover `length` with overlaps of at least 64, the slack spread over the overlaps
    // in whole latents.
    if (kTile >= length)
        return {{0, length, 0}};
    uint32_t n = (length + kTile - 1) / kTile;
    while (int64_t(kTile) * n - int64_t(kTileOverlap) * (n - 1) - int64_t(length) < 0)
        ++n;
    std::vector<uint32_t> overlaps(n - 1, kTileOverlap);
    const uint32_t remaining = kTile * n - kTileOverlap * (n - 1) - length;
    for (uint32_t i = 0; i < remaining / kPatch; ++i)
        overlaps[i % (n - 1)] += kPatch;
    std::vector<Tile> t{{0, kTile, 0}};
    for (uint32_t i = 1; i < n; ++i)
        t.push_back({t.back().start + kTile - overlaps[i - 1], kTile, overlaps[i - 1]});
    return t;
}

void H3Vae::vit(std::span<const float> voxels, uint32_t f, uint32_t th, uint32_t tw, std::vector<float>& out)
{
    const uint32_t N0 = f * th * tw, N = N0 + kRegisters + 1, Tg = round_up(N, 256), D = dim_;
    const uint32_t q_pad = round_up(N, kAttnRows), k_pad = round_up(N, 128);
    const uint32_t A = axes(), R = rot(), O = this->out(), H_ = heads(), F = ffn_;
    if (voxels.size() != uint64_t(N0) * kLatent)
        throw std::runtime_error("vit: voxels of the tile's shape");
    MTL::Buffer* in = scratch("v_in", uint64_t(N0) * kEmbedCols * 4);
    MTL::Buffer* x = scratch("v_x", uint64_t(Tg) * D * 4);
    MTL::Buffer* xn = scratch("v_xn", uint64_t(Tg) * D * 4);
    MTL::Buffer* qkv = scratch("v_qkv", uint64_t(Tg) * 3 * D * 4);
    MTL::Buffer* qh = scratch("v_qh", uint64_t(H_) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("v_kh", uint64_t(H_) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("v_vh", uint64_t(H_) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("v_att", uint64_t(Tg) * D * 4);
    MTL::Buffer* y = scratch("v_y", uint64_t(Tg) * D * 4);
    MTL::Buffer* gu = scratch("v_gu", uint64_t(Tg) * 2 * F * 4);
    MTL::Buffer* ffm = scratch("v_ffm", uint64_t(Tg) * F * 4);
    MTL::Buffer* o = scratch("v_o", uint64_t(Tg) * O * 4);
    MTL::Buffer* cs = scratch("v_cs", uint64_t(N) * (R / 2) * 8);
    {
        auto* p = static_cast<float*>(in->contents());
        for (uint32_t t = 0; t < N0; ++t) {
            std::copy_n(voxels.begin() + uint64_t(t) * kLatent, kLatent, p + uint64_t(t) * kEmbedCols);
            std::fill_n(p + uint64_t(t) * kEmbedCols + kLatent, kEmbedCols - kLatent, 0.0f);
        }
    }
    std::memset(x->contents(), 0, x->length());
    // Rotary angles: each axis' coordinate centred on [-1, 1) times 2 pi times 8 frequencies of base 100, the time
    // axis' pairs first where there is one, then the height's, then the width's; the register and zero tokens at
    // the origin.
    {
        auto* c = static_cast<float*>(cs->contents());
        float inv[8];
        for (uint32_t j = 0; j < 8; ++j)
            inv[j] = 1.0f / std::pow(kRopeTheta, float(j) / 8.0f);
        const uint32_t sizes[3] = {A == 3 ? f : th, A == 3 ? th : tw, tw};
        for (uint32_t t = 0; t < N; ++t) {
            float coord[3] = {0, 0, 0};
            if (t < N0) {
                const uint32_t idx3[3] = {t / (th * tw), t / tw % th, t % tw};
                for (uint32_t a = 0; a < A; ++a) {
                    const uint32_t idx = idx3[a + 3 - A];
                    coord[a] = 2.0f * ((float(idx) + 0.5f) / float(sizes[a])) - 1.0f;
                }
            }
            for (uint32_t a = 0; a < A; ++a)
                for (uint32_t j = 0; j < 8; ++j) {
                    const float ang = 2.0f * std::numbers::pi_v<float> * coord[a] * inv[j];
                    const uint64_t at = (uint64_t(t) * (R / 2) + a * 8 + j) * 2;
                    c[at] = std::cos(ang);
                    c[at + 1] = std::sin(ang);
                }
        }
    }
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    proj(enc, embed_, in, 0, x, 0, N0);
    bias(enc, x, embed_b_, D, N0);
    {
        const uint32_t n = kRegisters * D;
        dispatch_at(enc, "copy_f32", {registers_, Arg{x, uint64_t(N0) * D * 4}}, &n, sizeof(n), MTL::Size(n, 1, 1),
                    MTL::Size(256, 1, 1));
    }
    for (uint32_t b = 0; b < blocks_.size(); ++b) {
        const Block& B = blocks_[b];
        cur_layer_ = b;
        norm(enc, x, &B.norm1, xn, N, D, D, kEps);
        proj(enc, B.qkv, xn, 0, qkv, 0, Tg);
        bias(enc, qkv, B.qkv_b, 3 * D, N);
        // Each head's q, k and v side by side in a row; q and k RMS-normed (no scale) with their first `rot` values
        // rotated.
        heads_partial(enc, qkv, 3 * D, 0, 3 * kHeadDim, kHeadDim, R, H_, N, q_pad, cs, &ones_, kEps, qh);
        heads_partial(enc, qkv, 3 * D, kHeadDim, 3 * kHeadDim, kHeadDim, R, H_, N, k_pad, cs, &ones_, kEps, kh);
        heads_partial(enc, qkv, 3 * D, 2 * kHeadDim, 3 * kHeadDim, kHeadDim, 0, H_, N, k_pad, cs, nullptr, 0.0f, vh);
        if (k_pad > N) {
            const uint32_t c[4] = {kHeadDim, k_pad, N, 0};
            for (MTL::Buffer* hb : {kh, vh})
                dispatch(enc, "heads_clear", {hb}, c, sizeof(c), MTL::Size(kHeadDim, k_pad - N, H_),
                         MTL::Size(64, 1, 1));
        }
        attention(enc, qh, kh, vh, att, N, q_pad, N, k_pad, H_, H_, 0, 0, kHeadDim);
        proj(enc, B.o, att, 0, y, 0, Tg);
        bias(enc, y, B.o_b, D, N);
        dispatch_at(enc, "scale_add", {x, y, B.scale1}, &D, sizeof(D), MTL::Size(D, N, 1), MTL::Size(256, 1, 1));
        norm(enc, x, &B.norm2, xn, N, D, D, kEps);
        proj(enc, B.w1, xn, 0, gu, 0, Tg);
        bias(enc, gu, B.w1_b, 2 * F, N);
        uint32_t nff = F;
        dispatch(enc, "swiglu_rows", {gu, ffm}, &nff, sizeof(nff), MTL::Size(nff, Tg, 1), MTL::Size(256, 1, 1));
        proj(enc, B.w2, ffm, 0, y, 0, Tg);
        bias(enc, y, B.w2_b, D, N);
        dispatch_at(enc, "scale_add", {x, y, B.scale2}, &D, sizeof(D), MTL::Size(D, N, 1), MTL::Size(256, 1, 1));
        probe(enc, "vit_block", int(b), x, uint64_t(N) * D);
    }
    {
        const LnArgs a{D, kEps};
        enc->setComputePipelineState(pso("layer_norm"));
        enc->setBuffer(x, 0, 0);
        enc->setBuffer(out_norm_w_.buf, out_norm_w_.off, 1);
        enc->setBuffer(out_norm_b_.buf, out_norm_b_.off, 2);
        enc->setBuffer(xn, 0, 3);
        enc->setBytes(&a, sizeof(a), 4);
        enc->dispatchThreadgroups(MTL::Size(N0, 1, 1), MTL::Size(256, 1, 1));
        after(enc, "layer_norm");
    }
    proj(enc, proj_out_, xn, 0, o, 0, Tg);
    bias(enc, o, proj_out_b_, O, N0);
    enc->endEncoding();
    finish(cb);
    // Each token's [3][patch_t][16][16] values to the frames [patch_t f][16 th][16 tw][3].
    const uint32_t H = kPatch * th, W = kPatch * tw, PT = patch_t_;
    out.resize(uint64_t(PT) * f * H * W * 3);
    const auto* src = static_cast<const float*>(o->contents());
    for (uint32_t t = 0; t < N0; ++t) {
        const uint32_t ft = t / (th * tw), ty = t / tw % th, tx = t % tw;
        const float* v = src + uint64_t(t) * O;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t pt = 0; pt < PT; ++pt)
                for (uint32_t py = 0; py < kPatch; ++py) {
                    float* dst = &out[((uint64_t(ft * PT + pt) * H + ty * kPatch + py) * W + tx * kPatch) * 3 + c];
                    const float* s = v + ((c * PT + pt) * kPatch + py) * kPatch;
                    for (uint32_t px = 0; px < kPatch; ++px)
                        dst[uint64_t(px) * 3] = s[px];
                }
    }
}

void H3Vae::decode_normalized(std::span<const float> latents, uint32_t f, uint32_t h, uint32_t w, std::vector<float>& out)
{
    if (!f || !h || !w || latents.size() != uint64_t(f) * h * w * kLatent)
        throw std::runtime_error(name_ + ": latents [frames][h][w][24]");
    const uint32_t H = h * kPatch, W = w * kPatch, pf = patch_t_ * f;
    // Denormalized and through the post-quant 1x1 convolution.
    std::vector<float> z(latents.size());
    for (uint64_t v = 0; v < latents.size() / kLatent; ++v) {
        float d[kLatent];
        for (uint32_t c = 0; c < kLatent; ++c)
            d[c] = latents[v * kLatent + c] * std_[c] + mean_[c];
        for (uint32_t o = 0; o < kLatent; ++o) {
            float s = pq_b_[o];
            for (uint32_t c = 0; c < kLatent; ++c)
                s += pq_w_[o * kLatent + c] * d[c];
            z[v * kLatent + o] = s;
        }
    }
    const auto rows = tiles(H), cols = tiles(W);
    out.resize(uint64_t(pf) * H * W * 3);
    std::vector<float> tile, vox;
    std::vector<std::vector<float>> row_tiles(cols.size()), prev_row(cols.size());
    // Each tile decoded alone (every frame at once), then stitched: a tile blended with the one above over their
    // overlap and with the one to its left (that one's own pixels), then cropped by the overlaps it shares with the
    // next.
    uint32_t y0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint32_t th = rows[i].size / kPatch, ly = rows[i].start / kPatch;
        const uint32_t keep_h = i + 1 < rows.size() ? rows[i].size - rows[i + 1].overlap : rows[i].size;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tw = cols[j].size / kPatch, lx = cols[j].start / kPatch;
            vox.resize(uint64_t(f) * th * tw * kLatent);
            for (uint32_t ft = 0; ft < f; ++ft)
                for (uint32_t yy = 0; yy < th; ++yy)
                    std::copy_n(&z[((uint64_t(ft) * h + ly + yy) * w + lx) * kLatent], uint64_t(tw) * kLatent,
                                &vox[((uint64_t(ft) * th + yy) * tw) * kLatent]);
            vit(vox, f, th, tw, row_tiles[j]);
        }
        uint32_t x0 = 0;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tH = rows[i].size, tW = cols[j].size;
            tile = row_tiles[j];
            auto at = [&](std::vector<float>& t, uint32_t fr, uint32_t yy, uint32_t xx) -> float* {
                return &t[((uint64_t(fr) * tH + yy) * tW + xx) * 3];
            };
            if (i > 0) { // the tile above: its bottom rows fade into this one's top rows
                const uint32_t ov = rows[i].overlap;
                for (uint32_t fr = 0; fr < pf; ++fr)
                    for (uint32_t yy = 0; yy < ov; ++yy) {
                        const float wb = float(yy) / float(ov), wa = 1.0f - wb;
                        const float* a = at(prev_row[j], fr, tH - ov + yy, 0);
                        float* b = at(tile, fr, yy, 0);
                        for (uint32_t k = 0; k < tW * 3; ++k)
                            b[k] = a[k] * wa + b[k] * wb;
                    }
            }
            if (j > 0) { // the tile to the left, as decoded
                const uint32_t ov = cols[j].overlap;
                for (uint32_t fr = 0; fr < pf; ++fr)
                    for (uint32_t yy = 0; yy < tH; ++yy)
                        for (uint32_t xx = 0; xx < ov; ++xx) {
                            const float wb = float(xx) / float(ov), wa = 1.0f - wb;
                            const float* a = at(row_tiles[j - 1], fr, yy, tW - ov + xx);
                            float* b = at(tile, fr, yy, xx);
                            for (uint32_t k = 0; k < 3; ++k)
                                b[k] = a[k] * wa + b[k] * wb;
                        }
            }
            const uint32_t keep_w = j + 1 < cols.size() ? tW - cols[j + 1].overlap : tW;
            for (uint32_t fr = 0; fr < pf; ++fr)
                for (uint32_t yy = 0; yy < keep_h; ++yy)
                    std::copy_n(at(tile, fr, yy, 0), uint64_t(keep_w) * 3, &out[((uint64_t(fr) * H + y0 + yy) * W + x0) * 3]);
            x0 += keep_w;
        }
        y0 += keep_h;
        std::swap(prev_row, row_tiles);
    }
}

std::vector<float> H3Vae::decode(std::span<const float> latents, uint32_t f, uint32_t h, uint32_t w)
{
    std::vector<float> rgb;
    decode_normalized(latents, f, h, w, rgb);
    for (uint64_t i = 0; i < rgb.size(); ++i)
        rgb[i] = std::clamp(rgb[i] * kPixelStd[i % 3] + kPixelMean[i % 3], 0.0f, 1.0f);
    return rgb;
}

void H3Vae::load_encoder(bool temporal)
{
    if (temporal ? enc3d_ : enc_)
        return;
    const auto f = open_files(weights_);
    static constexpr uint32_t kMult[] = {1, 2, 2, 4, 4, 8};
    auto conv3 = [&](const std::string& name, uint32_t stride = 1) {
        const std::string n[1] = {name};
        Conv c = load_conv(f, n, 3, temporal);
        c.mode = Pad::reflect;
        c.stride = stride;
        c.pad = stride == 2 ? 0 : 1; // the downsample pads below and right only
        return c;
    };
    auto conv1 = [&](const std::string& name) {
        const std::string n[1] = {name};
        return load_conv(f, n, 1);
    };
    auto e = std::make_unique<Encoder>();
    e->in = conv3("encoder.conv_in");
    uint32_t cin = kEncCh;
    for (uint32_t l = 0; l < 6; ++l) {
        EncLevel L;
        const uint32_t cout = kEncCh * kMult[l];
        for (uint32_t b = 0; b < 2; ++b) {
            const std::string p = std::format("encoder.down.{}.block.{}.", l, b);
            EncRes r;
            r.g1 = upload_f32(f, p + "norm1.weight");
            r.b1 = upload_f32(f, p + "norm1.bias");
            r.c1 = conv3(p + "conv1");
            r.g2 = upload_f32(f, p + "norm2.weight");
            r.b2 = upload_f32(f, p + "norm2.bias");
            r.c2 = conv3(p + "conv2");
            r.shortcut = static_cast<bool>(f.find(p + "nin_shortcut.weight"));
            if (r.shortcut)
                r.sc = conv1(p + "nin_shortcut");
            if (r.c1.cin != cin || r.c2.cout != cout)
                throw std::runtime_error(p + ": unexpected widths");
            cin = cout;
            L.res.push_back(std::move(r));
        }
        if (l < 4) {
            L.down = true;
            L.down_conv = conv3(std::format("encoder.down.{}.downsample.conv", l), 2);
        }
        e->levels.push_back(std::move(L));
    }
    e->out_g = upload_f32(f, "encoder.norm_out.weight");
    e->out_b = upload_f32(f, "encoder.norm_out.bias");
    e->out = conv3("encoder.conv_out");
    e->quant = conv1("quant_conv");
    if (e->out.cout != 2 * kLatent || e->quant.cout != 2 * kLatent)
        throw std::runtime_error(name_ + ": the encoder does not end in 48 moments");
    (temporal ? enc3d_ : enc_) = std::move(e);
    log::info("{} encoder loaded{} ({:.2f} GB resident in all)", name_, temporal ? " with its time taps" : "",
              double(dense_bytes_) / 1e9);
}

void H3Vae::encode_clip_tile(std::span<const float> pixels, uint32_t T, uint32_t th, uint32_t tw, std::vector<float>& out,
                             uint32_t& T_out)
{
    static constexpr uint32_t kTimeDown[6] = {1, 2, 2, 1, 1, 1};
    const uint32_t rows = th * tw;
    MTL::Buffer* x = scratch("e3_x", uint64_t(T) * rows * 3 * 4);
    std::memcpy(x->contents(), pixels.data(), pixels.size() * 4);
    const uint64_t cap = uint64_t(T) * rows * kEncCh * 4;
    MTL::Buffer* h = scratch("e3_h", cap);
    MTL::Buffer* t1 = scratch("e3_t1", cap);
    MTL::Buffer* t2 = scratch("e3_t2", cap);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    const Encoder& E = *enc3d_;
    conv3d(enc, E.in, x, T, th, tw, 1, h);
    uint32_t hh = th, ww = tw, Tc = T;
    // Group norms are per frame (the checkpoint's time-isolated norm).
    auto gn = [&](MTL::Buffer* src, const Tensor& g, const Tensor& b, MTL::Buffer* dst, uint32_t c) {
        for (uint32_t f = 0; f < Tc; ++f)
            group_norm(enc, src, g, b, dst, hh * ww, c, true, 1e-6f, uint64_t(f) * hh * ww * c * 4,
                       uint64_t(f) * hh * ww * c * 4);
    };
    for (size_t l = 0; l < E.levels.size(); ++l) {
        const EncLevel& L = E.levels[l];
        for (const EncRes& r : L.res) {
            const uint32_t n = Tc * hh * ww;
            gn(h, r.g1, r.b1, t1, r.c1.cin);
            conv3d(enc, r.c1, t1, Tc, hh, ww, 1, t2);
            gn(t2, r.g2, r.b2, t2, r.c2.cin);
            conv3d(enc, r.c2, t2, Tc, hh, ww, 1, t1);
            if (r.shortcut) {
                conv(enc, r.sc, h, Tc * hh, ww, t2); // 1x1: every frame's pixels as rows
                add(enc, t2, t1, h, n * r.c2.cout);
            } else {
                add(enc, h, t1, h, n * r.c2.cout);
            }
        }
        if (L.down) {
            conv3d(enc, L.down_conv, h, Tc, hh, ww, kTimeDown[l], t1);
            std::swap(h, t1);
            Tc = frames_out(Tc, kTimeDown[l]);
            hh /= 2;
            ww /= 2;
        }
    }
    gn(h, E.out_g, E.out_b, t1, E.out.cin);
    conv3d(enc, E.out, t1, Tc, hh, ww, 1, t2);
    conv(enc, E.quant, t2, Tc * hh, ww, t1);
    enc->endEncoding();
    finish(cb);
    T_out = Tc;
    out.resize(uint64_t(Tc) * hh * ww * 2 * kLatent);
    std::memcpy(out.data(), t1->contents(), out.size() * 4);
}

void H3Vae::video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w, std::vector<float>& out,
                          uint32_t& latent_frames)
{
    constexpr uint32_t kClip = 17, kDrop = 3;
    if (frames < 2 || !h || !w || h % kPatch || w % kPatch || rgb.size() != uint64_t(frames) * h * w * 3)
        throw std::runtime_error(name_ + ": frames [F][h][w][3] with h and w multiples of 16");
    load_encoder(true);
    const uint32_t padded = (frames + kClip - 1) / kClip * kClip, chunks = padded / kClip;
    const auto rows = tiles(h), cols = tiles(w);
    const uint32_t lh = h / kPatch, lw = w / kPatch, C = 2 * kLatent;
    std::vector<float> all; // [chunk latents][lh][lw][C] concatenated
    std::vector<float> pix, tile, mom;
    std::vector<std::vector<float>> row_tiles(cols.size()), prev_row(cols.size());
    uint32_t total = 0;
    for (uint32_t ci = 0; ci < chunks; ++ci) {
        std::vector<float> canvas; // this clip's stitched moments [Tc][lh][lw][C]
        uint32_t Tc = 0, y0 = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            const uint32_t tH = rows[i].size, lth = tH / kPatch;
            for (size_t j = 0; j < cols.size(); ++j) {
                const uint32_t tW = cols[j].size;
                pix.resize(uint64_t(kClip) * tH * tW * 3);
                for (uint32_t f = 0; f < kClip; ++f) {
                    const uint32_t src = std::min(ci * kClip + f, frames - 1); // the last frame repeated past the end
                    for (uint32_t yy = 0; yy < tH; ++yy)
                        for (uint32_t xx = 0; xx < tW; ++xx)
                            for (uint32_t c = 0; c < 3; ++c) {
                                const uint8_t v = rgb[(((uint64_t(src) * h) + rows[i].start + yy) * w + cols[j].start + xx) * 3 + c];
                                pix[((uint64_t(f) * tH + yy) * tW + xx) * 3 + c] = (float(v) / 255.0f - kPixelMean[c]) / kPixelStd[c];
                            }
                }
                encode_clip_tile(pix, kClip, tH, tW, row_tiles[j], Tc);
            }
            if (canvas.empty())
                canvas.assign(uint64_t(Tc) * lh * lw * C, 0.0f);
            uint32_t x0 = 0;
            for (size_t j = 0; j < cols.size(); ++j) {
                const uint32_t ltw = cols[j].size / kPatch;
                tile = row_tiles[j];
                auto at = [&](std::vector<float>& t, uint32_t f, uint32_t yy, uint32_t xx) -> float* {
                    return &t[((uint64_t(f) * lth + yy) * ltw + xx) * C];
                };
                if (i > 0) {
                    const uint32_t ov = rows[i].overlap / kPatch;
                    for (uint32_t f = 0; f < Tc; ++f)
                        for (uint32_t yy = 0; yy < ov; ++yy) {
                            const float wb = float(yy) / float(ov), wa = 1.0f - wb;
                            const float* a = at(prev_row[j], f, lth - ov + yy, 0);
                            float* b = at(tile, f, yy, 0);
                            for (uint32_t k = 0; k < ltw * C; ++k)
                                b[k] = a[k] * wa + b[k] * wb;
                        }
                }
                if (j > 0) {
                    const uint32_t ov = cols[j].overlap / kPatch;
                    for (uint32_t f = 0; f < Tc; ++f)
                        for (uint32_t yy = 0; yy < lth; ++yy)
                            for (uint32_t xx = 0; xx < ov; ++xx) {
                                const float wb = float(xx) / float(ov), wa = 1.0f - wb;
                                const float* a = at(row_tiles[j - 1], f, yy, ltw - ov + xx);
                                float* b = at(tile, f, yy, xx);
                                for (uint32_t k = 0; k < C; ++k)
                                    b[k] = a[k] * wa + b[k] * wb;
                            }
                }
                const uint32_t keep_h = i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
                const uint32_t keep_w = j + 1 < cols.size() ? ltw - cols[j + 1].overlap / kPatch : ltw;
                for (uint32_t f = 0; f < Tc; ++f)
                    for (uint32_t yy = 0; yy < keep_h; ++yy)
                        std::copy_n(at(tile, f, yy, 0), uint64_t(keep_w) * C,
                                    &canvas[((uint64_t(f) * lh + y0 + yy) * lw + x0) * C]);
                x0 += keep_w;
            }
            y0 += i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
            prev_row.swap(row_tiles);
        }
        all.insert(all.end(), canvas.begin(), canvas.end());
        total += Tc;
    }
    if (total <= kDrop)
        throw std::runtime_error(name_ + ": too few latent frames");
    latent_frames = total - kDrop;
    out.assign(all.begin(), all.begin() + uint64_t(latent_frames) * lh * lw * C);
}

void H3Vae::encode_video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                                 std::vector<float>& mean, std::vector<float>& sd)
{
    std::vector<float> m;
    uint32_t F = 0;
    video_moments(rgb, frames, h, w, m, F);
    const uint64_t n = m.size() / (2 * kLatent);
    mean.resize(n * kLatent);
    sd.resize(n * kLatent);
    for (uint64_t v = 0; v < n; ++v)
        for (uint32_t c = 0; c < kLatent; ++c) {
            mean[v * kLatent + c] = m[v * 2 * kLatent + c];
            sd[v * kLatent + c] = std::exp(0.5f * std::clamp(m[v * 2 * kLatent + kLatent + c], -30.0f, 20.0f));
        }
}

std::vector<float> H3Vae::encode_video(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                                       std::span<const float> noise)
{
    std::vector<float> mean, sd;
    encode_video_moments(rgb, frames, h, w, mean, sd);
    const uint32_t lh = h / kPatch, lw = w / kPatch, F = uint32_t(mean.size() / (uint64_t(lh) * lw * kLatent));
    if (!noise.empty() && noise.size() != mean.size())
        throw std::runtime_error(name_ + ": the noise is not [frames][h / 16][w / 16][24]");
    // torch draws the sample as [24][F][h][w]; the latents here are [F][h][w][24].
    const std::vector<float> drawn = noise.empty() ? torch_randn(mean.size(), kEncodeSeed) : std::vector<float>();
    std::vector<float> z(mean.size());
    for (uint32_t c = 0; c < kLatent; ++c)
        for (uint32_t f = 0; f < F; ++f)
            for (uint32_t y = 0; y < lh; ++y)
                for (uint32_t x = 0; x < lw; ++x) {
                    const uint64_t i = ((uint64_t(f) * lh + y) * lw + x) * kLatent + c;
                    const float e = noise.empty() ? drawn[((uint64_t(c) * F + f) * lh + y) * lw + x] : noise[i];
                    const float v = float(__fp16(mean[i] + sd[i] * e));
                    z[i] = (v - mean_[c]) / std_[c];
                }
    return z;
}

void H3Vae::encode_tile(std::span<const float> pixels, uint32_t th, uint32_t tw, std::vector<float>& out)
{
    const uint32_t rows = th * tw;
    MTL::Buffer* x = scratch("e_x", uint64_t(rows) * 3 * 4);
    std::memcpy(x->contents(), pixels.data(), pixels.size() * 4);
    // Channels-last activations; the widest level is the first (128 channels at full resolution).
    const uint64_t cap = uint64_t(rows) * kEncCh * 4;
    MTL::Buffer* h = scratch("e_h", cap);
    MTL::Buffer* t1 = scratch("e_t1", cap);
    MTL::Buffer* t2 = scratch("e_t2", cap);
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    conv(enc, enc_->in, x, th, tw, h);
    uint32_t hh = th, ww = tw;
    for (size_t l = 0; l < enc_->levels.size(); ++l) {
        const EncLevel& L = enc_->levels[l];
        for (const EncRes& r : L.res) {
            const uint32_t n = hh * ww;
            group_norm(enc, h, r.g1, r.b1, t1, n, r.c1.cin, true);
            conv(enc, r.c1, t1, hh, ww, t2);
            group_norm(enc, t2, r.g2, r.b2, t2, n, r.c2.cin, true);
            conv(enc, r.c2, t2, hh, ww, t1);
            if (r.shortcut) {
                conv(enc, r.sc, h, hh, ww, t2);
                add(enc, t2, t1, h, n * r.c2.cout);
            } else {
                add(enc, h, t1, h, n * r.c2.cout);
            }
        }
        if (L.down) {
            conv(enc, L.down_conv, h, hh, ww, t1);
            std::swap(h, t1);
            hh /= 2;
            ww /= 2;
        }
        probe(enc, "enc_level", int(l), h, uint64_t(hh) * ww * (L.down ? L.down_conv.cout : L.res.back().c2.cout));
    }
    const uint32_t n = hh * ww;
    group_norm(enc, h, enc_->out_g, enc_->out_b, t1, n, enc_->out.cin, true);
    conv(enc, enc_->out, t1, hh, ww, t2);
    conv(enc, enc_->quant, t2, hh, ww, t1);
    enc->endEncoding();
    finish(cb);
    out.resize(uint64_t(n) * 2 * kLatent);
    std::memcpy(out.data(), t1->contents(), out.size() * 4);
}

void H3Vae::moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& out)
{
    if (!h || !w || h % kPatch || w % kPatch || rgb.size() != uint64_t(h) * w * 3)
        throw std::runtime_error(name_ + ": a frame [h][w][3] with h and w multiples of 16");
    load_encoder(false);
    const auto rows = tiles(h), cols = tiles(w);
    const uint32_t lh = h / kPatch, lw = w / kPatch, C = 2 * kLatent;
    out.assign(uint64_t(lh) * lw * C, 0.0f);
    std::vector<float> pix, tile;
    std::vector<std::vector<float>> row_tiles(cols.size()), prev_row(cols.size());
    uint32_t y0 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint32_t tH = rows[i].size, lth = tH / kPatch;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t tW = cols[j].size;
            pix.resize(uint64_t(tH) * tW * 3);
            for (uint32_t yy = 0; yy < tH; ++yy)
                for (uint32_t xx = 0; xx < tW; ++xx)
                    for (uint32_t c = 0; c < 3; ++c) {
                        const uint8_t v = rgb[((uint64_t(rows[i].start + yy) * w) + cols[j].start + xx) * 3 + c];
                        pix[(uint64_t(yy) * tW + xx) * 3 + c] = (float(v) / 255.0f - kPixelMean[c]) / kPixelStd[c];
                    }
            encode_tile(pix, tH, tW, row_tiles[j]);
        }
        // Stitched as the reference stitches: each tile blended over its overlap with the raw tile above and the raw
        // one to its left, then cropped by the overlaps it shares with the next tiles.
        uint32_t x0 = 0;
        for (size_t j = 0; j < cols.size(); ++j) {
            const uint32_t ltw = cols[j].size / kPatch;
            tile = row_tiles[j];
            auto at = [&](std::vector<float>& t, uint32_t yy, uint32_t xx) -> float* { return &t[(uint64_t(yy) * ltw + xx) * C]; };
            if (i > 0) {
                const uint32_t ov = rows[i].overlap / kPatch;
                for (uint32_t yy = 0; yy < ov; ++yy) {
                    const float wb = float(yy) / float(ov), wa = 1.0f - wb;
                    const float* a = at(prev_row[j], lth - ov + yy, 0);
                    float* b = at(tile, yy, 0);
                    for (uint32_t k = 0; k < ltw * C; ++k)
                        b[k] = a[k] * wa + b[k] * wb;
                }
            }
            if (j > 0) {
                const uint32_t ov = cols[j].overlap / kPatch;
                for (uint32_t yy = 0; yy < lth; ++yy)
                    for (uint32_t xx = 0; xx < ov; ++xx) {
                        const float wb = float(xx) / float(ov), wa = 1.0f - wb;
                        const float* a = at(row_tiles[j - 1], yy, ltw - ov + xx);
                        float* b = at(tile, yy, xx);
                        for (uint32_t k = 0; k < C; ++k)
                            b[k] = a[k] * wa + b[k] * wb;
                    }
            }
            const uint32_t keep_h = i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
            const uint32_t keep_w = j + 1 < cols.size() ? ltw - cols[j + 1].overlap / kPatch : ltw;
            for (uint32_t yy = 0; yy < keep_h; ++yy)
                std::copy_n(at(tile, yy, 0), uint64_t(keep_w) * C, &out[((uint64_t(y0) + yy) * lw + x0) * C]);
            x0 += keep_w;
        }
        y0 += i + 1 < rows.size() ? lth - rows[i + 1].overlap / kPatch : lth;
        prev_row.swap(row_tiles);
    }
}

void H3Vae::encode_moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& mean,
                           std::vector<float>& sd)
{
    std::vector<float> m;
    moments(rgb, h, w, m);
    const uint64_t n = m.size() / (2 * kLatent);
    mean.resize(n * kLatent);
    sd.resize(n * kLatent);
    for (uint64_t v = 0; v < n; ++v)
        for (uint32_t c = 0; c < kLatent; ++c) {
            mean[v * kLatent + c] = m[v * 2 * kLatent + c];
            sd[v * kLatent + c] = std::exp(0.5f * std::clamp(m[v * 2 * kLatent + kLatent + c], -30.0f, 20.0f));
        }
}

std::vector<float> H3Vae::torch_randn(size_t n, uint32_t seed)
{
    if (n < 16)
        throw std::runtime_error("torch_randn: at least 16 values");
    std::mt19937 g(seed);
    auto uniform = [&] { return float(g() & 0xffffffu) * (1.0f / 16777216.0f); };
    std::vector<float> d(n);
    for (float& v : d)
        v = uniform();
    auto box = [&](size_t j) {
        for (size_t k = 0; k < 8; ++k) {
            const float u1 = 1.0f - d[j + k], u2 = d[j + k + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1)), theta = 2.0f * std::numbers::pi_v<float> * u2;
            d[j + k] = radius * std::cos(theta);
            d[j + k + 8] = radius * std::sin(theta);
        }
    };
    for (size_t j = 0; j + 16 <= n; j += 16)
        box(j);
    if (n % 16) { // the tail is redrawn from fresh uniforms
        for (size_t k = n - 16; k < n; ++k)
            d[k] = uniform();
        box(n - 16);
    }
    return d;
}

std::vector<float> H3Vae::encode(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::span<const float> noise)
{
    std::vector<float> mean, sd;
    encode_moments(rgb, h, w, mean, sd);
    const uint32_t lh = h / kPatch, lw = w / kPatch;
    if (!noise.empty() && noise.size() != mean.size())
        throw std::runtime_error(name_ + ": the noise is not [h / 16][w / 16][24]");
    // torch draws the sample channel-major ([24][h][w]); the latents here are [h][w][24].
    const std::vector<float> drawn = noise.empty() ? torch_randn(mean.size(), kEncodeSeed) : std::vector<float>();
    std::vector<float> z(mean.size());
    for (uint32_t c = 0; c < kLatent; ++c)
        for (uint32_t y = 0; y < lh; ++y)
            for (uint32_t x = 0; x < lw; ++x) {
                const uint64_t i = (uint64_t(y) * lw + x) * kLatent + c;
                const float e = noise.empty() ? drawn[(uint64_t(c) * lh + y) * lw + x] : noise[i];
                const float v = float(__fp16(mean[i] + sd[i] * e)); // the sample kept to fp16
                z[i] = (v - mean_[c]) / std_[c];
            }
    return z;
}

} // namespace mako::models
