#pragma once

#include "models/runtime.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::models {

// The Qwen text encoders the image and video models condition on, from the original BF16 safetensors (the language
// model's tensors under model.language_model., language_model. or model.): Qwen3-VL's text part (grouped key / value
// heads, QK norms, rotary positions the same on the three axes for text), and Qwen3.5's hybrid of gated delta rule
// layers and gated attention. The weights live in this runtime alone, so an engine frees them by dropping it.
class QwenText final : public TransformerRuntime {
public:
    struct Options {
        float theta = 0;     // the rotary base where no config.json gives one (0: 5e6)
        uint32_t layers = 0; // keep only the first `layers` (0: all): a model read at a tapped layer needs no more
    };
    // The safetensors in `dir` (a directory, or one file), its config.json beside them when there is one.
    QwenText(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry, const Options& opts);
    QwenText(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);

    const char* name() const override { return "text encoder"; }

    // Rows [drop, n) of the last layer's output before the final norm ([n - drop][E]) or after it (normed), or with
    // taps, of the hidden states listed (0: the embeddings, i: layer i's output), [n - drop][taps][E].
    std::vector<float> encode(std::span<const int32_t> ids, uint32_t drop, std::span<const uint32_t> taps = {},
                              bool normed = false);
    uint32_t width() const { return E_; }
    uint32_t layers() const { return uint32_t(layers_.size()); }
    bool qwen35() const { return qwen35_; }

private:
    struct Layer {
        Tensor attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down;
        // A gated delta rule layer (Qwen3.5's linear attention): q | k | v in one matrix (q), its causal convolution,
        // the output gate z, b | a for the update and decay, the decay's parameters; o its output, q_norm the gated
        // norm's scale.
        bool linear = false;
        Tensor conv, z, ba, a_log, dt_bias;
    };
    void load(const std::string& dir, const Options& opts);
    // A gated delta rule layer and a gated attention layer (Qwen3.5) from xn, into y.
    void linear_layer(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* xn, MTL::Buffer* y, uint32_t T);
    void gated_attention(MTL::ComputeCommandEncoder*& enc, const Layer& L, MTL::Buffer* xn, MTL::Buffer* y, uint32_t T,
                         MTL::Buffer* cs);
    // Rows of the token embedding table (BF16, read from the file).
    void embed(std::span<const int32_t> ids, float* out) const;

    safetensors::Files files_;
    safetensors::Files::Ref embd_;
    std::vector<Layer> layers_;
    Tensor norm_;
    uint32_t E_ = 0, heads_ = 0, kv_ = 0;
    float eps_ = 1e-6f, theta_ = 5e6f;
    // Qwen3.5: zero-centred norms, attention heads head_ wide with an output gate and the first rot_ values rotated,
    // and the delta rule's key and value heads (of 128).
    bool qwen35_ = false;
    uint32_t head_ = kHeadDim, rot_ = kHeadDim, key_heads_ = 0, value_heads_ = 0;
};

} // namespace mako::models
