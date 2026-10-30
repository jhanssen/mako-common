#pragma once

#include "gguf/gguf.hpp"
#include "mtl/context.hpp"
#include "mtl/gemm.hpp"
#include "mtl/gemv.hpp"

#include <chrono>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mako::gpu {

// How much the engine measures and reports: summary (machine, load, per-token averages), steps (every step too),
// layers (every step's per-layer GPU time too), kernels (GPU time per kernel instead of per layer). At the kernels
// level every dispatch gets its own timestamped encoder, which inflates kernel times severalfold for small kernels
// and about 2x for large ones; use it to find candidates, not to size them.
enum class Telemetry { off, summary, steps, layers, kernels };
std::optional<Telemetry> parse_telemetry(std::string_view s);

struct KernelTime {
    std::string kernel;
    double ms = 0;
    uint32_t count = 0;
};

// What a model's GPU code shares, whatever the model: resident weights and scratch buffers, pipelines, dispatch,
// command buffers, encoder timestamps, probes, and the GEMV / GEMM wrappers. Engines derive from it.
class Runtime {
public:
    struct Tensor {
        MTL::Buffer* buf = nullptr;
        uint64_t off = 0;
        gguf::Type type = gguf::Type::F32;
        std::vector<uint64_t> dims;
    };

    // Called after each named stage of a step with the fp32 values it produced (for parity checks).
    using Probe = std::function<void(const std::string& name, int layer, const float* data, size_t n)>;
    void set_probe(Probe probe) { probe_ = std::move(probe); }

    struct Stats {
        double gpu_wait_ms = 0, gpu_exec_ms = 0; // exec: GPU start to end, summed
        double encode_ms = 0, step_ms = 0;       // encode: step start to the first commit
    };
    const Stats& stats() const { return stats_; }

    // Bytes of all scratch buffers allocated so far.
    uint64_t scratch_bytes() const
    {
        uint64_t n = 0;
        for (const auto& [name, b] : scratch_)
            n += b->length();
        return n;
    }

protected:
    Runtime(mtl::Context& ctx, Telemetry telemetry);
    virtual ~Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Room for `bytes` in the resident weight buffers, for the caller to fill.
    Tensor reserve(uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims);
    // `bytes` at `offset` of a file into the resident weight buffers (BF16 becomes F32 unless keep_bf16: matrices for
    // the BF16 GEMM).
    Tensor upload_raw(int fd, uint64_t offset, uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims,
                      const std::string& what, bool keep_bf16 = false);

    MTL::Buffer* scratch(const std::string& name, uint64_t bytes);
    MTL::ComputePipelineState* pso(const std::string& fn);

    // proj() of at least this many tokens uses the GEMM (fewer: the GEMV, which reads each weight once for them and
    // gives every token the same result as alone).
    uint32_t gemm_min_tokens_ = 5;
    // batch > 1: that many matrices from w on, each its own y block of t rows; x_shared: all read the same x.
    void gemv(MTL::ComputeCommandEncoder*& enc, const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t batch = 1, uint32_t t = 1, bool x_shared = false);
    // b's rows follow a's in memory, same type and shape: one batched GEMV can run both.
    static bool adjacent(const Tensor& a, const Tensor& b);
    void proj(MTL::ComputeCommandEncoder*& enc, const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
              uint64_t y_off, uint32_t T, uint32_t ldy = 0);
    void dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                  const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads = true);
    void norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor* w, MTL::Buffer* y, uint32_t rows,
              uint32_t n, uint32_t wn, float eps, uint64_t x_off = 0, uint64_t y_off = 0, uint32_t ystride = 0);
    // Independent GEMVs over the same tokens in one dispatch (Gemv::encode_jobs), when every job's format has a
    // kernel there (gemv_jobs_ok); the table rows come from the step's jobs_table_.
    bool gemv_jobs_ok(std::span<const mtl::Gemv::Job> jobs) const;
    void gemv_jobs(MTL::ComputeCommandEncoder*& enc, std::span<const mtl::Gemv::Job> jobs);
    mtl::Gemv::Job gemv_job(const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y, uint64_t y_off,
                            uint32_t batch, uint32_t t, bool x_shared) const;
    // SwiGLU FFN of T rows of xn into out (limit > 0: the gate and up values clamped to it).
    void ffn_dense(MTL::ComputeCommandEncoder*& enc, const Tensor& gate, const Tensor& up, const Tensor& down,
                   MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T, uint64_t x_off = 0,
                   uint64_t out_off = 0);
    // A half copy of rows x cols of x (rows ld floats apart, 0: cols) for the GEMM, in one shared scratch buffer.
    MTL::Buffer* halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows, uint32_t cols,
                        uint32_t ld);

    // A compute encoder, timed as `section` of `layer` when per-layer telemetry is on.
    MTL::ComputeCommandEncoder* encoder(MTL::CommandBuffer* cb, uint32_t layer, int section,
                                        MTL::DispatchType type = MTL::DispatchTypeSerial);
    // At the kernels telemetry level, closes the encoder holding the dispatch just encoded, labelled `kernel`,
    // and opens the next one.
    void after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel);
    // Commits cb, waits for it (on_committed() runs in between) and accounts for it and the ones committed ahead.
    void finish(MTL::CommandBuffer* cb);
    void submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);
    // Commits what cb holds, so the GPU starts on it while the rest of the step is encoded into a new command
    // buffer (finish() accounts for both). Not while commit_ahead_ok() says no.
    void commit_ahead(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc);
    // finish(): after the commit, before waiting (work the GPU waits on the CPU for); after the wait.
    virtual void on_committed(MTL::CommandBuffer*) {}
    virtual void on_completed() {}
    virtual bool commit_ahead_ok() const { return true; }

    void probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
               uint64_t off = 0);
    void deliver_probes();

    bool timing() const { return !samples_.empty() && tel_ >= Telemetry::layers; }
    // Timestamp sample buffers for at least `want` samples.
    void add_samples(uint64_t want);
    // At the layers level: also time named parts of each layer (part()), with at least `samples` samples.
    void enable_part_timing(bool on, uint64_t samples);
    // At the layers level with part timing on: the dispatches from here to the next part() (or the end of the
    // section's encoder) are timed as `name`, within the current section.
    void part(MTL::ComputeCommandEncoder*& enc, std::string_view name);
    // With part timing on: names the encoder just opened as part `name` (an encoder opened outside part()).
    void name_part(std::string_view name)
    {
        if (part_timing_ && !timed_.empty())
            timed_.back().kernel = name;
    }
    // The step's encoder timestamps: each timed section's GPU time to `section_ms` (layer, section, ms), named
    // parts and kernels summed into `parts` and `kernels`. GPU ticks are converted with the CPU clock over the step,
    // from timestamp pairs taken at its start (cpu0, gpu0) and now.
    void resolve_times(uint64_t cpu0, uint64_t gpu0, const std::function<void(uint32_t, int, double)>& section_ms,
                       std::vector<KernelTime>& parts, std::vector<KernelTime>& kernels);

    // MAKO_REPEAT=<label substrings, comma-separated>:<n> encodes matching GEMVs, norms and helper dispatches n times, so the extra
    // step time measures their cost in place (they all recompute their output from their inputs). Repeats read
    // weights the cache may still hold; n = 0 leaves them out instead (the output is then meaningless).
    int repeats(std::string_view label) const
    {
        if (repeat_n_ == 1)
            return 1;
        for (size_t p = 0; p <= repeat_label_.size();) { // comma-separated label substrings
            const size_t e = std::min(repeat_label_.find(',', p), repeat_label_.size());
            if (e > p && label.find(std::string_view(repeat_label_).substr(p, e - p)) != std::string_view::npos)
                return repeat_n_;
            p = e + 1;
        }
        return 1;
    }

    mtl::Context& ctx_;
    std::unique_ptr<mtl::Gemv> gemv_;
    std::unique_ptr<mtl::Gemm> gemm_;
    mtl::Ref<MTL::Library> lib_;
    std::map<std::string, mtl::Ref<MTL::ComputePipelineState>> psos_;

    std::vector<mtl::Ref<MTL::Buffer>> weight_chunks_;
    uint64_t chunk_used_ = 0;
    uint64_t dense_bytes_ = 0;
    std::map<std::string, mtl::Ref<MTL::Buffer>> scratch_;

    Probe probe_;
    std::chrono::steady_clock::time_point step_start_;
    Telemetry tel_ = Telemetry::off; // the level in effect (prefill runs kernel-level telemetry as summary)
    bool part_timing_ = false;
    std::string repeat_label_;
    int repeat_n_ = 1;
    std::string last_pso_; // the kernel a direct dispatch runs, for repeats()
    MTL::CommandBuffer* cur_cb_ = nullptr;
    std::vector<MTL::CommandBuffer*> ahead_; // committed by commit_ahead, waited for in finish()
    mtl::Ref<MTL::Buffer> jobs_table_;       // multi-GEMV dispatches' job tables of the step, jobs_off_ used so far
    uint64_t jobs_off_ = 0;
    uint32_t cur_layer_ = 0;
    bool encoding_ = false;
    Stats stats_;

private:
    struct Timed {
        uint32_t layer;
        int section; // >= 0: the engine's section of the layer; -1 one kernel
        NS::UInteger start, end;
        std::string kernel; // kernels level; empty: the encoder held no dispatch
    };
    std::vector<mtl::Ref<MTL::CounterSampleBuffer>> samples_; // kSamplesPerBuffer each; index / that picks one
    std::vector<Timed> timed_;
    NS::UInteger next_sample_ = 0;
    struct PendingProbe {
        std::string name;
        int layer;
        mtl::Ref<MTL::Buffer> copy;
        size_t n;
    };
    std::vector<PendingProbe> pending_probes_;
};

} // namespace mako::gpu
