#pragma once

#include "models/conv.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mako::models {

// MiniMax-H3's VAE, the video model's and the image model's: a causal 3D CNN encoder (group norms, SiLU and
// reflect-padded 3x3 convolutions in six levels of two residual blocks, a stride-2 downsample after the first four,
// 16x down to 48 moments, a 1x1 quant convolution) run over one frame (its last time taps) or over clips of frames,
// and a ViT decoder over latent voxels (a token each plus four register tokens and a zero token; pre-norm blocks of
// heads of 64 with unscaled RMS norms on q and k and rotary positions of 16 frequencies an axis at base 100 over the
// token's time, height and width, or its height and width alone; residuals scaled per channel; a gated SiLU FFN; a
// layer norm and a projection to the token's 16x16 patch of pixel frames). The video decoder is 36 blocks of 2048
// with a token four frames deep, the image decoder 24 blocks of 1024 with a token one frame deep: the weights'
// shapes tell them apart. Both run in 256-pixel tiles whose overlaps of at least 64 are blended, as the models were
// released. Pixels are ImageNet-normalized RGB inside; the weights F16 here (F32 in the files).
class H3Vae : public ConvRuntime {
public:
    // The weights (a safetensors file with the model's own tensor names), the latent statistics (a config.json with
    // latents_mean / latents_std), and the name the log calls it by.
    H3Vae(const std::string& weights, const std::string& config, mtl::Context& ctx,
          gpu::Telemetry telemetry = gpu::Telemetry::off, const std::string& name = "VAE");
    const char* name() const override { return name_.c_str(); }
    static constexpr uint32_t kLatent = 24, kPatch = 16, kHeadDim = 64, kRegisters = 4;
    static constexpr uint32_t kTile = 256, kTileOverlap = 64;
    static constexpr uint32_t kEncCh = 128, kEncodeSeed = 42; // the encoder's base width; the posterior's seed
    static constexpr float kPixelMean[3] = {0.485f, 0.456f, 0.406f}, kPixelStd[3] = {0.229f, 0.224f, 0.225f};
    // The decoder's shape, from its weights.
    uint32_t dim() const { return dim_; }
    uint32_t heads() const { return dim_ / kHeadDim; }
    uint32_t ffn() const { return ffn_; }
    uint32_t patch_t() const { return patch_t_; } // pixel frames a token holds: 4 (the video model), 1 (the image)
    uint32_t axes() const { return patch_t_ > 1 ? 3 : 2; } // rotary axes: time, height, width; or height, width
    uint32_t rot() const { return 16 * axes(); }
    uint32_t out() const { return 3 * patch_t_ * kPatch * kPatch; }
    uint32_t blocks() const { return uint32_t(blocks_.size()); }
    std::span<const float> latents_mean() const { return mean_; }
    std::span<const float> latents_std() const { return std_; }

    // Normalized latents [f][h][w][24] to pixels [patch_t f][16 h][16 w][3] in [0, 1]: the tiles of every frame
    // decoded in one pass each and stitched, a tile blended with the one above over their overlap and with the one
    // to its left (that one's own pixels), then cropped by the overlaps it shares with the next. A video's clips
    // are the caller's (the video engine decodes seven latent frames at a time and cross-fades the clips).
    std::vector<float> decode(std::span<const float> latents, uint32_t f, uint32_t h, uint32_t w);
    // The same into `out`, in the decoder's own (ImageNet-normalized) space.
    void decode_normalized(std::span<const float> latents, uint32_t f, uint32_t h, uint32_t w, std::vector<float>& out);

    // An image [h][w][3] of 8-bit RGB (h and w multiples of 16) to its normalized latents [h / 16][w / 16][24]: the
    // encoder on the single frame (its last time taps), in tiles blended as the decoder's, the posterior sampled
    // with `noise` ([h / 16][w / 16][24]) or, without one, as the released video pipeline draws it (torch's generator
    // at seed 42, reproduced), the sample rounded to fp16. The encoder's weights load on first use.
    std::vector<float> encode(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::span<const float> noise = {});
    // The posterior's mean and standard deviation, [h / 16][w / 16][24] each.
    void encode_moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& mean,
                        std::vector<float>& sd);
    // A video's frames [F][h][w][3] (F = 17 n + 5 at 24 fps) to its normalized latents [5 n + 2][h / 16][w / 16]
    // [24]: the encoder over 17-frame clips (the frames padded by repeating the last one, the last three latents
    // dropped), tiled and sampled as encode(); the three time taps load on first use.
    std::vector<float> encode_video(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                                    std::span<const float> noise = {});
    void encode_video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                              std::vector<float>& mean, std::vector<float>& sd);
    // torch's CPU randn(n, generator=manual_seed(seed)): mt19937's 24-bit uniforms through Box-Muller over blocks of
    // 16 (matches to about 1e-6; n at least 16).
    static std::vector<float> torch_randn(size_t n, uint32_t seed);
    // Tiles over `length` pixels: each tile's start and the overlap with the one before it (0 for the first).
    struct Tile {
        uint32_t start, size, overlap;
    };
    static std::vector<Tile> tiles(uint32_t length);

private:
    struct Block {
        Tensor norm1, norm2, qkv, qkv_b, o, o_b, w1, w1_b, w2, w2_b, scale1, scale2;
    };
    // The ViT over a tile's voxels [f][th][tw][24] (denormalized, after the post-quant projection): pixels
    // [patch_t f][16 th][16 tw][3] in the decoder's own space.
    void vit(std::span<const float> voxels, uint32_t f, uint32_t th, uint32_t tw, std::vector<float>& out);
    std::string name_, weights_;
    uint32_t dim_ = 0, ffn_ = 0, patch_t_ = 0;
    Tensor embed_, embed_b_, registers_, ones_, out_norm_w_, out_norm_b_, proj_out_, proj_out_b_;
    std::vector<Block> blocks_;
    std::vector<float> mean_, std_, pq_w_, pq_b_; // latent statistics; the post-quant 1x1 convolution
    // The CNN encoder: conv_in, 6 levels of 2 residual blocks (group norm, SiLU, 3x3, reflect padded; a 1x1 shortcut
    // where the width changes) with a stride-2 downsample after the first 4, norm_out, conv_out to 48 moments, the
    // 1x1 quant convolution.
    struct EncRes {
        Tensor g1, b1, g2, b2;
        Conv c1, c2, sc;
        bool shortcut = false;
    };
    struct EncLevel {
        std::vector<EncRes> res;
        bool down = false;
        Conv down_conv;
    };
    struct Encoder {
        Conv in, out, quant;
        Tensor out_g, out_b;
        std::vector<EncLevel> levels;
    };
    std::unique_ptr<Encoder> enc_, enc3d_; // single-frame weights (the last time taps) and all three taps
    void load_encoder(bool temporal);
    // The 3D encoder over one clip tile [T][th][tw][3] (ImageNet-normalized) to moments [T'][th / 16][tw / 16][48].
    void encode_clip_tile(std::span<const float> pixels, uint32_t T, uint32_t th, uint32_t tw, std::vector<float>& out,
                          uint32_t& T_out);
    // A whole frame stack's moments [F'][h / 16][w / 16][48], tiles blended, clips concatenated, the tail dropped.
    void video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w, std::vector<float>& out,
                       uint32_t& latent_frames);
    // One tile of ImageNet-normalized pixels [th][tw][3] to its moments [th / 16][tw / 16][48] (mean, log variance).
    void encode_tile(std::span<const float> pixels, uint32_t th, uint32_t tw, std::vector<float>& moments);
    // The whole frame's moments, tiles blended.
    void moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& out);
};

} // namespace mako::models
