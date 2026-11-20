#include "models/qwen_text.hpp"

#include "gguf/read_pool.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mako::models {

namespace {

struct HeadsPartialArgs {
    uint32_t ld, col0, stride, dim, rot, n_pad;
    float eps;
};
struct AttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

glz::generic read_json_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (const auto err = glz::read_json(j, text))
        throw std::runtime_error(std::format("{}: {}", path, glz::format_error(err, text)));
    return j;
}

} // namespace

QwenText::QwenText(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry, const Options& opts)
    : TransformerRuntime(ctx, telemetry), files_(open_files(dir))
{
    load(dir, opts);
}

QwenText::QwenText(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry)
    : QwenText(dir, ctx, telemetry, Options{})
{
}

void QwenText::load(const std::string& dir, const Options& opts)
{
    // eps, the rotary base and the model type from the encoder's config, where it has one beside the weights.
    double rot_share = 1.0;
    if (opts.theta > 0)
        theta_ = opts.theta;
    if (std::filesystem::exists(dir + "/config.json")) {
        auto j = read_json_file(dir + "/config.json");
        const auto& t = j.contains("text_config") ? j["text_config"] : j;
        if (t.contains("rms_norm_eps") && t["rms_norm_eps"].is_number())
            eps_ = float(t["rms_norm_eps"].get_number());
        if (t.contains("rope_theta") && t["rope_theta"].is_number())
            theta_ = float(t["rope_theta"].get_number());
        const bool rp = t.contains("rope_parameters");
        if (rp && t["rope_parameters"].contains("rope_theta"))
            theta_ = float(t["rope_parameters"]["rope_theta"].get_number());
        qwen35_ = t.contains("model_type") && t["model_type"].is_string() &&
                  t["model_type"].get_string().starts_with("qwen3_5");
        if (qwen35_ && rp && t["rope_parameters"].contains("partial_rotary_factor"))
            rot_share = t["rope_parameters"]["partial_rotary_factor"].get_number();
    }
    // The language model's tensors, under the full model's prefix or the bare one's.
    std::string p;
    for (const char* prefix : {"model.language_model.", "language_model.", "model."})
        if (files_.find(std::string(prefix) + "embed_tokens.weight")) {
            p = prefix;
            break;
        }
    embd_ = files_.find(p + "embed_tokens.weight");
    if (!embd_ || embd_.tensor->type != gguf::Type::BF16)
        throw std::runtime_error("text encoder: embed_tokens missing or not BF16");
    E_ = uint32_t(embd_.tensor->dims[0]);
    // Qwen3.5 stores its norms' scales zero-centred.
    auto norm_w = [&](const std::string& name) {
        return qwen35_ ? upload_plus_one(files_, name) : upload(files_, name, false);
    };
    for (uint32_t i = 0; !opts.layers || i < opts.layers; ++i) {
        const std::string l = std::format("{}layers.{}.", p, i);
        if (!files_.find(l + "input_layernorm.weight"))
            break;
        Layer t;
        t.attn_norm = norm_w(l + "input_layernorm.weight");
        if (files_.find(l + "linear_attn.in_proj_qkv.weight")) {
            const std::string a = l + "linear_attn.";
            t.linear = true;
            t.q = upload(files_, a + "in_proj_qkv.weight", true);
            t.z = upload(files_, a + "in_proj_z.weight", true);
            const std::string ba[2] = {a + "in_proj_b.weight", a + "in_proj_a.weight"};
            t.ba = upload_rows(files_, ba);
            t.conv = upload(files_, a + "conv1d.weight", false);
            t.a_log = upload(files_, a + "A_log", false);
            t.dt_bias = upload(files_, a + "dt_bias", false);
            t.q_norm = upload(files_, a + "norm.weight", false);
            t.o = upload(files_, a + "out_proj.weight", true);
        } else {
            t.q = upload(files_, l + "self_attn.q_proj.weight", true);
            t.k = upload(files_, l + "self_attn.k_proj.weight", true);
            t.v = upload(files_, l + "self_attn.v_proj.weight", true);
            t.o = upload(files_, l + "self_attn.o_proj.weight", true);
            t.q_norm = norm_w(l + "self_attn.q_norm.weight");
            t.k_norm = norm_w(l + "self_attn.k_norm.weight");
        }
        t.ffn_norm = norm_w(l + "post_attention_layernorm.weight");
        t.gate = upload(files_, l + "mlp.gate_proj.weight", true);
        t.up = upload(files_, l + "mlp.up_proj.weight", true);
        t.down = upload(files_, l + "mlp.down_proj.weight", true);
        layers_.push_back(std::move(t));
    }
    if (layers_.empty())
        throw std::runtime_error("text encoder: no layers");
    if (opts.layers && layers_.size() < opts.layers)
        throw std::runtime_error(std::format("text encoder: {} layers, {} asked for", layers_.size(), opts.layers));
    if (files_.find(p + "norm.weight"))
        norm_ = norm_w(p + "norm.weight");
    const auto attn = std::ranges::find_if(layers_, [](const Layer& t) { return !t.linear; });
    if (attn == layers_.end())
        throw std::runtime_error("text encoder: no attention layer");
    head_ = uint32_t(attn->q_norm.dims[0]);
    if (head_ != kHeadDim && !(qwen35_ && head_ == 256))
        throw std::runtime_error(std::format("text encoder: {}-wide heads", head_));
    // Qwen3.5's queries carry an output gate per head: [heads][query, gate].
    heads_ = uint32_t(attn->q.dims[1] / (head_ * (qwen35_ ? 2 : 1)));
    kv_ = uint32_t(attn->k.dims[1] / head_);
    rot_ = uint32_t(head_ * rot_share);
    for (const Layer& t : layers_)
        if (t.linear) {
            value_heads_ = uint32_t(t.a_log.dims[0]);
            key_heads_ = uint32_t((t.q.dims[1] - uint64_t(value_heads_) * 128) / 256);
            if (t.q_norm.dims[0] != 128 || t.z.dims[1] != uint64_t(value_heads_) * 128 || !key_heads_ ||
                value_heads_ % key_heads_ || t.conv.dims[0] > 8)
                throw std::runtime_error("text encoder: delta rule heads not 128 wide");
            break;
        }
}

void QwenText::embed(std::span<const int32_t> ids, float* out) const
{
    const uint64_t vocab = embd_.tensor->dims[1], row = uint64_t(E_) * 2;
    std::vector<uint16_t> raw(E_);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= vocab)
            throw std::runtime_error("token out of range");
        gguf::read_exact(embd_.file->fd(), raw.data(), row, embd_.tensor->offset + uint64_t(ids[i]) * row);
        auto* o = reinterpret_cast<uint32_t*>(out + i * E_);
        for (uint32_t j = 0; j < E_; ++j)
            o[j] = uint32_t(raw[j]) << 16;
    }
}

std::vector<float> QwenText::encode(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps,
                                    bool normed, std::span<const Vision> vision)
{
    const uint32_t T = uint32_t(ids.size()), E = E_, kv = kv_ * kHeadDim, qw = heads_ * kHeadDim;
    if (T <= drop)
        throw std::runtime_error("the prompt is all template");
    uint32_t layers = uint32_t(layers_.size());
    if (!taps.empty()) {
        layers = *std::max_element(taps.begin(), taps.end());
        if (layers > layers_.size())
            throw std::runtime_error("text encoder: a tapped layer past its last");
    }
    const uint32_t nt = taps.empty() ? 1 : uint32_t(taps.size());
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    MTL::Buffer* x = scratch("t_x", uint64_t(T) * E * 4);
    MTL::Buffer* xn = scratch("t_xn", uint64_t(T) * E * 4);
    MTL::Buffer* q = scratch("t_q", uint64_t(T) * qw * 4);
    MTL::Buffer* k = scratch("t_k", uint64_t(T) * kv * 4);
    MTL::Buffer* v = scratch("t_v", uint64_t(T) * kv * 4);
    MTL::Buffer* qh = scratch("t_qh", uint64_t(heads_) * q_pad * kHeadDim * 2);
    MTL::Buffer* kh = scratch("t_kh", uint64_t(kv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* vh = scratch("t_vh", uint64_t(kv_) * k_pad * kHeadDim * 2);
    MTL::Buffer* att = scratch("t_att", uint64_t(T) * qw * 4);
    MTL::Buffer* y = scratch("t_y", uint64_t(T) * E * 4);
    const uint32_t pairs = rot_ / 2;
    MTL::Buffer* cs = scratch("t_cs", uint64_t(T) * pairs * 8);
    MTL::Buffer* out = scratch("t_out", uint64_t(T) * nt * E * 4);
    embed(ids, static_cast<float*>(x->contents()));
    // Every token's position on the three rotary axes (time, height, width): a text token's the same on all three,
    // which makes the multi-axis rotation the plain one; an image's tokens take their row and column. With
    // interleaved sections of 24, 20 and 20 pairs, pair i turns by the time axis when i % 3 == 0 or i >= 60, the
    // height axis when i % 3 == 1, the width axis when i % 3 == 2 (Qwen3-VL's M-RoPE; the plain rotation otherwise).
    std::vector<std::array<uint32_t, 3>> pos(T);
    {
        uint32_t cur = 0, t = 0;
        size_t vi = 0;
        while (t < T) {
            if (vi < vision.size() && vision[vi].first == t) {
                const Vision& v = vision[vi++];
                if (t + uint64_t(v.rows) * v.cols > T || v.embeds.size() != uint64_t(v.rows) * v.cols * E)
                    throw std::runtime_error("text encoder: an image's tokens do not fit the sequence");
                if (v.deepstack.size() != kDeepstack)
                    throw std::runtime_error("text encoder: an image needs its three DeepStack features");
                std::memcpy(static_cast<float*>(x->contents()) + uint64_t(t) * E, v.embeds.data(), v.embeds.size() * 4);
                for (uint32_t k = 0; k < v.rows * v.cols; ++k, ++t)
                    pos[t] = {cur, cur + k / v.cols, cur + k % v.cols};
                cur += std::max(v.rows, v.cols);
                continue;
            }
            pos[t++] = {cur, cur, cur};
            ++cur;
        }
        if (vi != vision.size())
            throw std::runtime_error("text encoder: an image starts past the sequence or out of order");
    }
    const bool mrope = !vision.empty() && !qwen35_;
    auto* c = static_cast<float*>(cs->contents());
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < pairs; ++i) {
            const float inv = 1.0f / std::pow(theta_, float(2 * i) / float(rot_));
            const uint32_t axis = mrope && i < 60 ? i % 3 : 0;
            const float ang = float(pos[t][axis]) * inv;
            c[(uint64_t(t) * pairs + i) * 2] = std::cos(ang);
            c[(uint64_t(t) * pairs + i) * 2 + 1] = std::sin(ang);
        }
    // The DeepStack features, added after the first layers.
    MTL::Buffer* ds = nullptr;
    if (!vision.empty()) {
        uint64_t total = 0;
        for (const Vision& v : vision)
            total += uint64_t(v.rows) * v.cols * E;
        ds = scratch("t_ds", total * kDeepstack * 4);
        auto* d = static_cast<float*>(ds->contents());
        for (uint32_t layer = 0; layer < kDeepstack; ++layer)
            for (const Vision& v : vision) {
                if (v.deepstack[layer].size() != uint64_t(v.rows) * v.cols * E)
                    throw std::runtime_error("text encoder: a DeepStack feature of the wrong size");
                std::memcpy(d, v.deepstack[layer].data(), v.deepstack[layer].size() * 4);
                d += v.deepstack[layer].size();
            }
    }

    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
    auto deepstack = [&](uint32_t layer) {
        if (!ds || layer >= kDeepstack)
            return;
        uint64_t off = 0;
        for (uint32_t l = 0; l < layer; ++l)
            for (const Vision& v : vision)
                off += uint64_t(v.rows) * v.cols * E;
        for (const Vision& v : vision) {
            const uint32_t n = v.rows * v.cols * E;
            dispatch_at(enc, "add", {Arg{x, uint64_t(v.first) * E * 4}, Arg{ds, off * 4}, Arg{x, uint64_t(v.first) * E * 4}},
                        &n, sizeof(n), MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
            off += n;
        }
    };
    // The first token's outlier channels grow to thousands and cancel in the last layers: activations in bfloat
    // alone leave its final state a few percent off.
    bf16_split_ = true;
    // Hidden state i (0: the embeddings) into its column of the taps' rows.
    auto tap = [&](uint32_t i) {
        for (uint32_t j = 0; j < taps.size(); ++j)
            if (taps[j] == i) {
                const uint32_t a[2] = {E, nt * E};
                enc->setComputePipelineState(pso("copy_rows"));
                enc->setBuffer(x, 0, 0);
                enc->setBuffer(out, uint64_t(j) * E * 4, 1);
                enc->setBytes(a, sizeof(a), 2);
                enc->dispatchThreads(MTL::Size(E, T, 1), MTL::Size(256, 1, 1));
                after(enc, "copy_rows");
            }
    };
    tap(0);
    for (uint32_t il = 0; il < layers; ++il) {
        const Layer& L = layers_[il];
        cur_layer_ = il;
        norm(enc, x, &L.attn_norm, xn, T, E, E, eps_);
        if (qwen35_) {
            if (L.linear)
                linear_layer(enc, L, xn, y, T);
            else
                gated_attention(enc, L, xn, y, T, cs);
            add(enc, x, y, x, T * E);
            norm(enc, x, &L.ffn_norm, xn, T, E, E, eps_);
            ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
            add(enc, x, y, x, T * E);
            deepstack(il);
            tap(il + 1);
            continue;
        }
        proj(enc, L.q, xn, 0, q, 0, T);
        proj(enc, L.k, xn, 0, k, 0, T);
        proj(enc, L.v, xn, 0, v, 0, T);
        norm(enc, q, &L.q_norm, q, T * heads_, kHeadDim, kHeadDim, eps_);
        norm(enc, k, &L.k_norm, k, T * kv_, kHeadDim, kHeadDim, eps_);
        heads(enc, q, qw, 0, heads_, T, q_pad, cs, 1, qh);
        heads(enc, k, kv, 0, kv_, T, k_pad, cs, 1, kh);
        heads(enc, v, kv, 0, kv_, T, k_pad, nullptr, 0, vh);
        attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, heads_, kv_, 0, UINT32_MAX);
        proj(enc, L.o, att, 0, y, 0, T);
        add(enc, x, y, x, T * E);
        norm(enc, x, &L.ffn_norm, xn, T, E, E, eps_);
        ffn_dense(enc, L.gate, L.up, L.down, xn, y, 0.0f, T);
        add(enc, x, y, x, T * E);
        deepstack(il);
        tap(il + 1);
    }
    if (normed) {
        if (!norm_.buf)
            throw std::runtime_error("text encoder: no final norm");
        norm(enc, x, &norm_, x, T, E, E, eps_);
    }
    bf16_split_ = false;
    enc->endEncoding();
    finish(cb);
    const auto* src = static_cast<const float*>((taps.empty() ? x : out)->contents());
    return std::vector<float>(src + uint64_t(drop) * nt * E, src + uint64_t(T) * nt * E);
}

void QwenText::linear_layer(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* xn, MTL::Buffer* y,
                            uint32_t T)
{
    const uint32_t KH = key_heads_, VH = value_heads_, width = (2 * KH + VH) * 128, vw = VH * 128;
    MTL::Buffer* m = scratch("g_m", uint64_t(T) * width * 4);
    MTL::Buffer* mc = scratch("g_mc", uint64_t(T) * width * 4);
    MTL::Buffer* z = scratch("g_z", uint64_t(T) * vw * 4);
    MTL::Buffer* ba = scratch("g_ba", uint64_t(T) * 2 * VH * 4);
    MTL::Buffer* q = scratch("g_q", uint64_t(T) * vw * 4);
    MTL::Buffer* k = scratch("g_k", uint64_t(T) * vw * 4);
    MTL::Buffer* v = scratch("g_v", uint64_t(T) * vw * 4);
    MTL::Buffer* g = scratch("g_g", uint64_t(T) * vw * 4);
    MTL::Buffer* beta = scratch("g_beta", uint64_t(T) * VH * 4);
    MTL::Buffer* o = scratch("g_o", uint64_t(T) * vw * 4);
    // Each layer's own state, zeroed now: the GPU runs the layers after all are encoded.
    MTL::Buffer* state = scratch(std::format("g_state{}", cur_layer_), uint64_t(VH) * 128 * 128 * 4);
    std::memset(state->contents(), 0, state->length());
    proj(enc, L.q, xn, 0, m, 0, T);
    proj(enc, L.z, xn, 0, z, 0, T);
    proj(enc, L.ba, xn, 0, ba, 0, T);
    const uint32_t ca[3] = {width, uint32_t(L.conv.dims[0]), T};
    enc->setComputePipelineState(pso("gdn_conv"));
    enc->setBuffer(m, 0, 0);
    enc->setBuffer(L.conv.buf, L.conv.off, 1);
    enc->setBuffer(mc, 0, 2);
    enc->setBytes(ca, sizeof(ca), 3);
    enc->dispatchThreads(MTL::Size(width, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "gdn_conv");
    struct {
        uint32_t k_heads, v_heads, conv_width;
        float eps;
    } pa{KH, VH, width, 1e-6f};
    enc->setComputePipelineState(pso("gdn_prep"));
    enc->setBuffer(mc, 0, 0);
    enc->setBuffer(ba, 0, 1);
    enc->setBuffer(L.a_log.buf, L.a_log.off, 2);
    enc->setBuffer(L.dt_bias.buf, L.dt_bias.off, 3);
    for (uint32_t i = 0; i < 5; ++i)
        enc->setBuffer(std::array{q, k, v, g, beta}[i], 0, 4 + i);
    enc->setBytes(&pa, sizeof(pa), 9);
    enc->dispatchThreadgroups(MTL::Size(VH, T, 1), MTL::Size(128, 1, 1));
    after(enc, "gdn_prep");
    // The delta rule with a decay per head: the per-channel scan with each head's decay on all of its channels.
    const uint32_t sa[6] = {128, T, vw, vw, VH, 0};
    enc->setComputePipelineState(pso("kda_scan"));
    for (uint32_t i = 0; i < 7; ++i)
        enc->setBuffer(std::array{q, k, v, g, beta, state, o}[i], 0, i);
    enc->setBytes(sa, sizeof(sa), 7);
    enc->setBuffer(state, 0, 8);
    enc->dispatchThreadgroups(MTL::Size(VH, 4, 1), MTL::Size(128, 1, 1));
    after(enc, "kda_scan");
    const struct {
        uint32_t heads;
        float eps;
    } ga{VH, eps_};
    enc->setComputePipelineState(pso("gated_rms"));
    enc->setBuffer(o, 0, 0);
    enc->setBuffer(z, 0, 1);
    enc->setBuffer(L.q_norm.buf, L.q_norm.off, 2);
    enc->setBytes(&ga, sizeof(ga), 3);
    enc->dispatchThreadgroups(MTL::Size(VH, T, 1), MTL::Size(128, 1, 1));
    after(enc, "gated_rms");
    proj(enc, L.o, o, 0, y, 0, T);
}

void QwenText::gated_attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* xn, MTL::Buffer* y,
                               uint32_t T, MTL::Buffer* cs)
{
    const uint32_t D = head_, H = heads_, KV = kv_, qw = 2 * H * D, kw = KV * D;
    const uint32_t q_pad = round_up(T, kAttnRows), k_pad = round_up(T, 128);
    MTL::Buffer* q = scratch("a_q", uint64_t(T) * qw * 4);
    MTL::Buffer* k = scratch("a_k", uint64_t(T) * kw * 4);
    MTL::Buffer* v = scratch("a_v", uint64_t(T) * kw * 4);
    MTL::Buffer* qh = scratch("a_qh", uint64_t(H) * q_pad * D * 2);
    MTL::Buffer* kh = scratch("a_kh", uint64_t(KV) * k_pad * D * 2);
    MTL::Buffer* vh = scratch("a_vh", uint64_t(KV) * k_pad * D * 2);
    MTL::Buffer* att = scratch("a_att", uint64_t(T) * H * D * 4);
    std::memset(qh->contents(), 0, qh->length());
    std::memset(kh->contents(), 0, kh->length());
    std::memset(vh->contents(), 0, vh->length());
    proj(enc, L.q, xn, 0, q, 0, T);
    proj(enc, L.k, xn, 0, k, 0, T);
    proj(enc, L.v, xn, 0, v, 0, T);
    heads_partial(enc, q, qw, 0, 2 * D, D, rot_, H, T, q_pad, cs, &L.q_norm, eps_, qh);
    heads_partial(enc, k, kw, 0, D, D, rot_, KV, T, k_pad, cs, &L.k_norm, eps_, kh);
    heads_partial(enc, v, kw, 0, D, D, 0, KV, T, k_pad, cs, nullptr, 0.0f, vh);
    attention(enc, qh, kh, vh, att, T, q_pad, T, k_pad, H, KV, 0, UINT32_MAX, D);
    const uint32_t ga[5] = {H, D, qw, D, 2 * D};
    dispatch(enc, "gate_heads", {att, q}, ga, sizeof(ga), MTL::Size(D, H, T), MTL::Size(64, 1, 1));
    proj(enc, L.o, att, 0, y, 0, T);
}

} // namespace mako::models
