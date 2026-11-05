#pragma once

#include "models/runtime.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::models {

// Qwen3-VL's vision tower (model.visual.* in the text encoder's safetensors): an image of h x w pixels (multiples of
// 32) to what its language model takes in place of the image's pad tokens: the 2x2-merged patch tokens
// [h / 32 * w / 32][5120], row-major over the merged grid, and the three DeepStack features of the same shape added to
// the hidden states after the language model's first three layers (QwenText::Vision). 16x16 patches embedded with two
// identical time taps, the learned 48x48 position grid resampled bilinearly (corners aligned), 27 pre-norm blocks
// with 2D axial rope over 72-wide heads and a GELU-tanh MLP, the mergers (LayerNorm, MLP with exact GELU).
class QwenVision final : public TransformerRuntime {
public:
    QwenVision(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);

    const char* name() const override { return "vision tower"; }

    static constexpr uint32_t kPatch = 16, kMerge = 2, kDim = 1152, kHeads = 16, kHeadDim = 72, kHeadPad = 128;
    static constexpr uint32_t kFfn = 4304, kFfnPad = 4352, kMerged = kDim * kMerge * kMerge, kOut = 5120, kGrid = 48;
    static constexpr uint32_t kDeepstack = 3;
    static constexpr uint32_t kDeepstackBlocks[kDeepstack] = {8, 16, 24}; // after these blocks
    static constexpr uint32_t kPixels = 3 * 2 * kPatch * kPatch;          // a patch's values

    struct Features {
        uint32_t rows = 0, cols = 0; // the merged grid
        std::vector<float> embeds;   // [rows * cols][5120]
        std::array<std::vector<float>, kDeepstack> deepstack;
    };
    // 8-bit RGB [h][w][3].
    Features encode(std::span<const uint8_t> rgb, uint32_t h, uint32_t w);

private:
    struct Block {
        Tensor ln1_w, ln1_b, qkv, qkv_b, proj, proj_b, ln2_w, ln2_b, fc1, fc1_b, fc2, fc2_b;
    };
    struct Merger {
        Tensor ln_w, ln_b, fc1, fc1_b, fc2, fc2_b;
    };
    void load(const std::string& dir);
    // A BF16 matrix [n][k] (floats in) in the resident weights.
    Tensor matrix(std::span<const float> w, uint32_t k, uint32_t n);
    Tensor vector(std::span<const float> v);
    void layer_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& w, const Tensor& b, MTL::Buffer* y,
                    uint32_t rows, uint32_t n);
    // merged [m][4608] (four patches a row) through a merger into out [m][5120].
    void merge(MTL::ComputeCommandEncoder*& enc, const Merger& mg, MTL::Buffer* x, uint32_t m, bool post,
               MTL::Buffer* t, MTL::Buffer* out);

    Tensor patch_w_, patch_b_;
    std::vector<float> pos_; // [48 * 48][1152]
    std::vector<Block> blocks_;
    Merger merger_;
    std::array<Merger, kDeepstack> deep_;
    float eps_ = 1e-6f, theta_ = 10000.0f;
};

} // namespace mako::models
