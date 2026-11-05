#pragma once

#include "gpu/runtime.hpp"
#include "safetensors/safetensors.hpp"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mako::models {

// A tensor's values as floats (F32, F16 or BF16).
std::vector<float> read_floats(const safetensors::File& f, const gguf::Tensor& t);

// What the transformer models share on top of the GPU runtime: their weights from safetensors files, attention
// heads in half tiles with rotary positions, flash attention, small activations, and LoRAs merged into (or kept
// beside) the matrices. An engine derives from it and names its matrices for the LoRA loader.
class TransformerRuntime : public gpu::Runtime {
public:
    // Adds a LoRA for good: W += scale * (alpha / rank) * B A for each matrix it adapts, merged (rounded back to
    // the weights' BF16 or F16) or, where the model can (keeps_loras_unmerged()), kept beside the weight and applied
    // on every pass; b += scale * diff_b for whole bias differences. merge unset: the model's way (unmerged where it
    // can). Keys as Diffusers, PEFT or ComfyUI name them, or Kohya's (lora_unet_ and the module path with
    // underscores); alpha from the adapter's .alpha tensors, else its PEFT config in the metadata, else the rank.
    // Returns the tensors adapted; throws on a file it cannot use (keys it does not know are an error).
    uint32_t add_lora(const std::string& path, float scale = 1.0f, std::optional<bool> merge = std::nullopt);
    // Whether LoRAs can stay unmerged (merge false).
    virtual bool keeps_loras_unmerged() const { return false; }
    // The model's name, for messages.
    virtual const char* name() const { return "model"; }

protected:
    TransformerRuntime(mtl::Context& ctx, gpu::Telemetry telemetry) : gpu::Runtime(ctx, telemetry) {}

    static constexpr uint32_t kHeadDim = 128;
    static constexpr uint32_t kAttnRows = 64; // queries a threadgroup of the attention kernel
    static uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

    // A tensor's values as floats.
    static std::vector<float> floats(const safetensors::Files& files, const std::string& name);
    // A tensor as it is (BF16 matrices stay BF16 for the BF16 GEMM when `matrix`; F32 matrices become BF16; other
    // BF16 tensors become F32).
    Tensor upload(const safetensors::Files& files, const std::string& name, bool matrix);
    // A tensor's values as F32 (from F32, F16 or BF16).
    Tensor upload_f32(const safetensors::Files& files, const std::string& name);
    // A BF16, F16 or F32 matrix as I8: int8 rows with a float scale each (the GEMM at 1.7x BF16's rate, half the
    // bytes; LoRAs merge into it by requantizing the rows). Stages the file's values in the scratch "i8_stage".
    Tensor upload_i8(const safetensors::Files& files, const std::string& name);
    // BF16 (or F32, rounded to BF16) matrices of one width, their rows one after another in one tensor; F16 ones
    // stay F16.
    Tensor upload_rows(const safetensors::Files& files, std::span<const std::string> names);
    // A matrix as F16 (from F32, F16 or BF16), for the direct F16 GEMM.
    Tensor upload_f16(const safetensors::Files& files, const std::string& name);
    // An RMS norm's scale stored zero-centred: weight + 1, as floats.
    Tensor upload_plus_one(const safetensors::Files& files, const std::string& name);

    // A kernel's buffer argument: a whole buffer, or a tensor at its offset in one.
    struct Arg {
        MTL::Buffer* buf;
        uint64_t off = 0;
        Arg(MTL::Buffer* b, uint64_t o = 0) : buf(b), off(o) {}
        Arg(const Tensor& t) : buf(t.buf), off(t.off) {}
    };
    // dispatch() with the arguments' offsets.
    void dispatch_at(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<Arg> args,
                     const void* bytes, size_t bytes_size, MTL::Size grid, MTL::Size tg, bool threads = true);

    void add(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, MTL::Buffer* z, MTL::Buffer* y, uint32_t n);
    void unary(MTL::ComputeCommandEncoder*& enc, const char* fn, MTL::Buffer* x, uint32_t n);
    // y[r][i] += b[i] over rows of n.
    void bias(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* y, const Tensor& b, uint32_t n, uint32_t rows,
              uint64_t y_off = 0);
    // Per-head q / k / v half tiles of T rows (rope: 0 none, 1 halves, 2 adjacent pairs; cs: the angle table), the
    // rows past T zeroed.
    void heads(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
               uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint32_t rope, MTL::Buffer* y);
    // heads() with each head RMS-normed first (times w); the rows past T are left as they are.
    void heads_norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t heads,
                    uint32_t T, uint32_t n_pad, MTL::Buffer* cs, uint64_t cs_off, uint32_t rope, const Tensor& w,
                    float eps, uint32_t row0, MTL::Buffer* y);
    // Heads of `dim` (at most 256) at col0 + h * stride of each row, RMS-normed (times w) when w is given, their
    // first `rot` values rotated as halves (i with i + rot / 2) by cs[t][i] (rot / 2 pairs a row); into half tiles
    // [heads][n_pad][dim].
    void heads_partial(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t ld, uint32_t col0, uint32_t stride,
                       uint32_t dim, uint32_t rot, uint32_t heads, uint32_t T, uint32_t n_pad, MTL::Buffer* cs,
                       const Tensor* w, float eps, MTL::Buffer* y, MTL::Buffer* map = nullptr);
    // Flash attention over half tiles q [heads][q_pad][dim], k and v [kv_heads][k_pad][dim] into o [n_q][heads *
    // dim] floats. block: positions from here on see each other both ways; below it, causal (0: all both ways).
    // dim 64, 128 or 256.
    void attention(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v, MTL::Buffer* o,
                   uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t heads, uint32_t kv_heads,
                   uint32_t q_pos0, uint32_t block, uint32_t dim = kHeadDim);
    // Block-sparse attention over 128-wide heads for a sequence whose first n_cubes * 64 rows are 64-row cubes of
    // video tokens (padding rows with valid 0) and whose rows [dense0, dense1) are the others: cube_select picks, per
    // head and query cube, the `topk` key cubes by the dot product of the cubes' mean q and k (cnt[c]: a cube's valid
    // rows), into sel [heads][n_cubes][topk] and nsel; attention_sparse then attends each cube's queries to those
    // cubes and the dense rows, the dense rows' queries to everything, and writes query row r's output to o row
    // map[r] (map[r] < 0: dropped). q, k, v as attention() takes them (q_pad a multiple of 64, k_pad of 128).
    void cube_select(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, uint32_t heads, uint32_t k_pad,
                     uint32_t n_cubes, MTL::Buffer* cnt, uint32_t topk, MTL::Buffer*& sel, MTL::Buffer*& nsel);
    void attention_sparse(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* q, MTL::Buffer* k, MTL::Buffer* v,
                          MTL::Buffer* o, uint32_t n_q, uint32_t q_pad, uint32_t n_k, uint32_t k_pad, uint32_t heads,
                          uint32_t kv_heads, uint32_t width, uint32_t n_cubes, uint32_t topk, uint32_t dense0,
                          uint32_t dense1, MTL::Buffer* sel, MTL::Buffer* nsel, MTL::Buffer* valid, MTL::Buffer* map);

    // The weight a LoRA module adapts: its matrix and the rows it has there (fused matrices share one); b_rot: the
    // adapter's up rows rotated onto them (row r takes up row (r + b_rot) % rows: a fused matrix whose halves the
    // adapter orders the other way round).
    struct LoraTarget {
        Tensor* w = nullptr;
        uint32_t row0 = 0, rows = 0, b_rot = 0;
    };
    // `module` with the file's prefixes stripped (transformer., diffusion_model., ...); none: not adapted.
    virtual LoraTarget lora_target(const std::string& /*module*/) { return {}; }
    // The F32 bias of `module`, for an adapter's whole bias difference (null: none).
    virtual Tensor* lora_bias(const std::string&) { return nullptr; }
    // A small matrix of `module` the model keeps on the CPU, [rows][cols] floats (null: none).
    virtual float* lora_cpu(const std::string&, uint32_t& /*rows*/, uint32_t& /*cols*/) { return nullptr; }
    // After a LoRA is merged (what the model computed from the weights before is stale).
    virtual void lora_merged() {}
    // Keeps W += s * B A for `module` unmerged, applied beside W on every pass (a: [rank][k], b: [rows][rank]);
    // false: merge it. For changes smaller than the weights' rounding, which merging into them would lose.
    virtual bool lora_side(const std::string& /*module*/, std::span<const float> /*a*/, std::span<const float> /*b*/,
                           uint32_t /*rank*/, float /*s*/)
    {
        return false;
    }
    // Set while a control adapter's LoRA is added (a file marked as one is otherwise refused): its input
    // projection's tensors (video_embedder.) are the model's to read.
    bool control_lora_ = false;
};

} // namespace mako::models
