#pragma once

#include "models/runtime.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mako::models {

// MiniMax-H3's diffusion transformer, the video model's and the image model's: 50 blocks of full self-attention over
// one packed sequence [text | conditions | audio | video], each row modulated by the AdaLN parameters of its
// (timestep, modality); a two-block token refiner on the projected text; one velocity head for the video rows and,
// where the model has an audio stream, one for the audio rows. From the original BF16 shards (rows of the fused q /
// k / v matrices interleaved per head) or a single-file checkpoint of the same names (rows stacked q, k, v),
// including the pruned ones whose time embedder and modulation projections are folded into an 8-coordinate curve
// table (adaln_t_table), as the ComfyUI and Civitai files are. The image model's checkpoint has no audio stream and
// one video head per step of its distilled schedule (step() takes the step's head): stacked in its own
// final_layer.video_out (the first whole, the others as offsets from it), or in a heads file beside the shards.
class H3 final : public TransformerRuntime {
public:
    // Which of the 50 blocks' matrices are kept as int8 rows with a scale each (the GEMM at 1.7x BF16's rate, half
    // the bytes; the activations quantized per row before each): a mask of these.
    enum Int8 : uint32_t { kInt8Qkv = 1, kInt8Out = 2, kInt8Fc1 = 4, kInt8Fc2 = 8, kInt8All = 15 };
    // "none", "all", or a comma list of qkv, out, fc1 and fc2; nullopt for anything else.
    static std::optional<uint32_t> parse_int8(std::string_view s);

    // The transformer's directory (its shards and config.json) or a single file; int8: the Int8 mask; heads: a file
    // of per-step video heads (video_out.weight [steps][96][5376] with its bias) in place of the checkpoint's own.
    H3(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off, uint32_t int8 = 0,
       const std::string& heads = "");

    // Block-sparse attention over the generated video rows (Video Sparse Attention's training-free form): the rows
    // in 64-row cubes of 4 latent frames x 4 x 4 patches, each cube's queries attending the share `ratio` of the
    // cubes a coarse pass over the cubes' mean q and k picks per head (at least one), and the text, conditioning and
    // audio rows; those rows' queries attend everything. 0: dense attention. Takes effect at the next begin().
    // local: the cubes every query cube keeps regardless of score: 0 none, 1 itself, 3 itself and the two a frame
    // group before and after it.
    void set_sparse(float ratio, uint32_t local = 1)
    {
        sparse_ = ratio;
        sparse_local_ = local;
    }
    float sparse() const { return sparse_; }
    static constexpr uint32_t kCube = 4; // latent frames, patch rows and patch columns of a cube

    const char* name() const override { return "MiniMax-H3"; }
    // Whether the checkpoint has the audio stream (the video model's).
    bool has_audio() const { return static_cast<bool>(audio_in_.buf); }
    // The video heads: one, or one per step of a distilled schedule.
    uint32_t heads() const { return uint32_t(video_out_.size()); }

    static constexpr uint32_t kDim = 5376, kHeads = 56, kText = 5120, kVideoPatch = 96, kAudioChannels = 32;
    static constexpr uint32_t kTimeFreq = 256, kTimeDim = 2688, kRot = 96, kRopeFreqs = 16;
    static constexpr uint32_t kCurve = 8, kCurvePoints = 1025; // a pruned checkpoint's curve table
    static constexpr float kRopeTheta = 10000.0f;
    // Modality tags, which index the AdaLN table with the timestep.
    static constexpr uint32_t kVideoTag = 0, kTextTag = 1, kAudioTag = 2;

    // The packed sequence of a generation: text rows; then the conditioning the task brings, fl2va's keyframes
    // (frames on the canvas) or ref2va's references in reading order (an image's rows, a soundtrack's audio rows, a
    // video's audio rows then its video rows, each on its own grid); then the generated audio rows (channel-major:
    // every latent of the left channel, then the right's) and the generated video rows (frame-major, then row-major
    // over 2x2 patches of the latent frame). The video rows step() takes are the conditioning ones first (keyframes,
    // then the references' in order), then the generated; the audio rows the references' first. An image is one
    // frame with no audio.
    struct Layout {
        uint32_t text = 0;
        uint32_t frames = 0;      // generated latent frames
        uint32_t h = 0, w = 0;    // latent height and width (even)
        uint32_t audio = 0;       // audio latents a channel (0 for a model without the audio stream)
        uint32_t cond_frames = 0; // keyframe conditioning frames before the generated ones (fl2va)
        std::vector<uint8_t> cond_last; // per conditioning frame: anchored at the last generated frame (else the first)
        struct Reference {
            enum class Kind : uint8_t { image, audio, video };
            Kind kind = Kind::image;
            uint32_t frames = 0, h = 0, w = 0; // image, video: latent frames (an image's 1), latent height and width (even)
            uint32_t audio = 0;                // audio, or a video's soundtrack: latents a channel (0: a silent video)
            uint32_t video_rows() const { return kind == Kind::audio ? 0 : frames * (h / 2) * (w / 2); }
            uint32_t audio_rows() const { return 2 * audio; }
        };
        std::vector<Reference> refs;    // ref2va's references in reading order (not together with keyframes)
        std::vector<uint8_t> text_tags; // a tag per text row (kTextTag, or kVideoTag for a vision block's); empty:
                                        // all text
        uint32_t rows_per_frame() const { return (h / 2) * (w / 2); }
        uint32_t cond_rows() const // conditioning video rows: the keyframes' and the references'
        {
            uint32_t n = cond_frames * rows_per_frame();
            for (const Reference& r : refs)
                n += r.video_rows();
            return n;
        }
        uint32_t cond_audio_rows() const
        {
            uint32_t n = 0;
            for (const Reference& r : refs)
                n += r.audio_rows();
            return n;
        }
        uint32_t video_rows() const { return cond_rows() + frames * rows_per_frame(); }
        uint32_t audio_rows() const { return cond_audio_rows() + 2 * audio; }
        uint32_t rows() const { return text + video_rows() + audio_rows(); }
    };
    // Every row of a layout's sequence: its rotary position (t, h, w) as the reference lays them out (the text on the
    // time axis at its row index, a shared clock continuing from there that every block advances: an image one unit,
    // audio one a latent, video 5/3 x (1, 4, 4, 4, 4) units a latent frame; the spatial axes centred on the unit
    // square of the block's own grid, times 32), its modality tag and its role; and the video and audio segments of
    // the sequence, each with its first sequence row and its first row in what step() takes.
    enum Role : uint8_t { kRoleText, kRoleCondVideo, kRoleCondAudio, kRoleAudio, kRoleVideo };
    struct Segment {
        bool audio;
        uint32_t row0, n, src;
    };
    struct Rows {
        std::vector<double> pos; // [rows][3]
        std::vector<uint8_t> tags, roles;
        std::vector<Segment> segments;
    };
    static Rows rows(const Layout& l);
    // Begins a generation: the text embeddings [text][5120] through the condition projection and the token refiner
    // (kept for every step), the layout's rotary positions and tags.
    void begin(std::span<const float> text, const Layout& layout);
    // One forward: the video rows [video_rows()][96] (conditioning rows first) and audio rows [audio_rows()][32]
    // (reference rows first) at noise levels t = 1 - sigma (t_video for the generated video rows and the text,
    // t_audio for the generated audio, t_cond for the conditioning video rows; reference audio rows are clean, at
    // kCondAudioT), their velocities out (conditioning and reference rows' zero). head: the video head of this step.
    struct Velocity {
        std::vector<float> video, audio;
    };
    static constexpr float kCondAudioT = 1.0f;
    Velocity step(std::span<const float> video, std::span<const float> audio, float t_video, float t_audio,
                  float t_cond = 0.999f, uint32_t head = 0);
    const Layout& layout() const { return layout_; }

protected:
    LoraTarget lora_target(const std::string& module) override;
    Tensor* lora_bias(const std::string& module) override;

private:
    struct Block {
        Tensor norm1, norm2, qkv, o, norm_q, norm_k, fc1, fc2;
        Tensor adaln, adaln_b; // the transformer blocks' modulation projection (none in the refiner's)
    };
    void load(const safetensors::Files& f, const std::string& heads);
    Block load_block(const safetensors::Files& f, const std::string& prefix, bool modulated);
    // The fused q / k / v matrix with its rows stacked [q; k; v] (the shards interleave them per head).
    Tensor upload_qkv(const safetensors::Files& f, const std::string& name, bool int8);
    // A BF16 matrix for the GEMM, or an F16 one as stored.
    Tensor upload_matrix(const safetensors::Files& f, const std::string& name);
    // A small F32 matrix with its rows padded to a multiple of 256, so it runs on the GEMM.
    Tensor upload_padded(const safetensors::Files& f, const std::string& name, uint32_t rows);
    // One video head of `n` rows from `w` [n][5376] (row-major floats) and its bias, the rows padded to kOutRows.
    void add_head(std::span<const float> w, std::span<const float> b, uint32_t n);
    // The video heads: the checkpoint's own final_layer.video_out (one head, or a distilled schedule's stacked: the
    // first whole, the others offsets from it), or the heads file's.
    void load_heads(const safetensors::Files& f, const std::string& heads);
    // Attention and the FFN of a block over `rows` rows of x (its first `rows` of Tg), modulated per row from mods
    // (null: the refiner's plain pre-norm block).
    void block(MTL::ComputeCommandEncoder*& enc, const Block& B, MTL::Buffer* x, uint32_t rows, uint32_t Tg,
               MTL::Buffer* mods, uint64_t mods_off, MTL::Buffer* off, MTL::Buffer* cs);

    std::vector<Block> blocks_, refiner_;
    Tensor refiner_norm_, cond_, cond_b_, video_in_, video_in_b_, audio_in_, audio_in_b_;
    Tensor t1_, t1_b_, t2_, t2_b_, final_adaln_, final_adaln_b_, final_norm_, audio_out_, audio_out_b_;
    std::vector<Tensor> video_out_, video_out_b_; // a head per distilled step (one for the video model)
    bool interleaved_ = false; // the file's q / k / v rows interleaved per head
    uint32_t int8_ = 0;        // the Int8 mask
    // A pruned checkpoint: the time embedder and the 2688-wide modulation inputs folded into an 8-coordinate curve
    // c(t), tabulated at 1025 points of t in [0, 1] (adaln_t_table) and read with linear interpolation; the blocks'
    // and the final layer's modulation projections then take the 8 coordinates (no SiLU).
    bool curve_ = false;
    std::vector<float> table_; // [1025][8]
    Layout layout_;
    Rows rows_;
    std::vector<uint32_t> tags_; // a modality tag per row
    uint32_t cond_rows_ = 0, S_ = 0;
    // Sparse attention: the attention's own row order [video cubes (padded) | text | conditions | audio], as a map
    // from its rows to the model's (-1: a padding row), each cube's valid rows, the dense rows' range.
    float sparse_ = 0.0f;
    uint32_t sparse_local_ = 1;
    struct Cubes {
        uint32_t n = 0, rows = 0, topk = 0, dense0 = 0, dense1 = 0; // cubes; rows in the attention's order
        uint32_t stride = 0;                                        // cubes a frame group apart
        MTL::Buffer *map = nullptr, *valid = nullptr, *cnt = nullptr;
    } cubes_;
};

} // namespace mako::models
