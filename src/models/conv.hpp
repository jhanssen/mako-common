#pragma once

#include "models/runtime.hpp"
#include "safetensors/safetensors.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::models {

// What the models' convolutional parts share (the image VAEs, MiniMax-H3's video VAE encoder): 3x3 and 1x1
// convolutions as GEMMs over channels-last images ([height][width][channels], a pixel a row) with F16 weights, their
// loading, and the group norm. A causal 3D convolution on a single frame is its last time tap's 2D one.
class ConvRuntime : public TransformerRuntime {
protected:
    ConvRuntime(mtl::Context& ctx, gpu::Telemetry telemetry) : TransformerRuntime(ctx, telemetry) {}

    // What a convolution's padding holds: zeros, or the image reflected without its edge (PyTorch's reflect).
    enum class Pad { zeros, reflect };
    // Weights F16 [cout][kpad] (cin * k * k columns, a 3x3's zero padded to a multiple of 64). A causal 3D
    // convolution's weights are its last time tap's (a single frame).
    struct Conv {
        Tensor w, b;
        uint32_t cin = 0, cout = 0, k = 3, kpad = 0, stride = 1;
        uint32_t pad = 1; // stride 2: padding above and left (below and right always 1)
        Pad mode = Pad::zeros;
        uint32_t kt = 1;  // time taps kept: 1 (a single frame's), or 3 (a causal 3D convolution over frames)
    };
    // `name`.weight and .bias (F32, F16 or BF16): a convolution (or a linear layer, as a 1x1), or several of one
    // input's with their outputs side by side. temporal: all three time taps of a 3D kernel (conv3d), else the last.
    Conv load_conv(const safetensors::Files& files, std::span<const std::string> names, uint32_t k,
                   bool temporal = false);
    // y = conv(x) over an h x w image (with stride 2, a 3x3 to h / 2 x w / 2).
    void conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t h, uint32_t w, MTL::Buffer* y);
    // y = conv(x) over T frames [T][h][w][cin], causal in time (two zero frames before the first), time stride ts (1
    // or 2: ceil(T / 2) frames out); the spatial part as conv(). A Conv loaded with temporal taps.
    void conv3d(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t T, uint32_t h, uint32_t w,
                uint32_t ts, MTL::Buffer* y);
    // Frames out of conv3d for T in at time stride ts.
    static uint32_t frames_out(uint32_t T, uint32_t ts) { return ts == 1 ? T : (T + 1) / 2; }
    // Group norm (32 groups) times g plus b over channels-last images [rows][c] (from x_off / y_off bytes in), then
    // SiLU when silu is set.
    void group_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor& g, const Tensor& b, MTL::Buffer* y,
                    uint32_t rows, uint32_t c, bool silu, float eps = 1e-6f, uint64_t x_off = 0, uint64_t y_off = 0);
};

} // namespace mako::models
