#include "gpu/runtime.hpp"

#include "gguf/read_pool.hpp"
#include "mtl/embedded.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <stdexcept>

namespace mako::gpu {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr uint64_t kAlign = 256;
constexpr uint64_t kChunkBytes = 1ull << 30;
constexpr NS::UInteger kSamplesPerBuffer = 4096; // the largest counter sample buffer is 32 KiB

struct NormArgs {
    uint32_t n, wn;
    float eps;
    uint32_t ystride;
};
struct SwigluArgs {
    uint32_t n;
    float limit;
};

} // namespace

std::optional<Telemetry> parse_telemetry(std::string_view s)
{
    if (s == "off")
        return Telemetry::off;
    if (s == "summary")
        return Telemetry::summary;
    if (s == "steps")
        return Telemetry::steps;
    if (s == "layers")
        return Telemetry::layers;
    if (s == "kernels")
        return Telemetry::kernels;
    return std::nullopt;
}

Runtime::Runtime(mtl::Context& ctx, Telemetry telemetry) : ctx_(ctx), tel_(telemetry)
{
    if (const char* r = std::getenv("MAKO_REPEAT")) {
        const std::string_view spec(r);
        const size_t colon = spec.rfind(':');
        if (colon == std::string_view::npos)
            throw std::runtime_error("MAKO_REPEAT wants <label substring>:<n>");
        repeat_label_ = spec.substr(0, colon);
        repeat_n_ = std::max(0, std::atoi(r + colon + 1));
    }
    gemv_ = std::make_unique<mtl::Gemv>(ctx_);
    gemm_ = std::make_unique<mtl::Gemm>(ctx_);
    lib_ = ctx_.load(mtl::kernels_metallib());
}

Runtime::Tensor Runtime::reserve(uint64_t bytes, gguf::Type type, std::vector<uint64_t> dims)
{
    if (weight_chunks_.empty() || chunk_used_ + bytes > weight_chunks_.back()->length()) {
        const uint64_t size = std::max(kChunkBytes, (bytes + kAlign - 1) / kAlign * kAlign);
        weight_chunks_.emplace_back(ctx_.device()->newBuffer(size, MTL::ResourceStorageModeShared));
        if (!weight_chunks_.back())
            throw std::runtime_error("weight buffer allocation failed");
        chunk_used_ = 0;
    }
    Tensor d;
    d.buf = weight_chunks_.back().get();
    d.off = chunk_used_;
    d.type = type;
    d.dims = std::move(dims);
    chunk_used_ += (bytes + kAlign - 1) / kAlign * kAlign;
    dense_bytes_ += bytes;
    return d;
}

Runtime::Tensor Runtime::upload_raw(int fd, uint64_t offset, uint64_t bytes, gguf::Type type,
                                    std::vector<uint64_t> dims, const std::string& what)
{
    const bool bf16 = type == gguf::Type::BF16;
    if (type != gguf::Type::F32 && !bf16 && !mtl::Gemv::supports(type))
        throw std::runtime_error(std::format("{}: unsupported type {}", what, gguf::traits(type).name));
    Tensor d = reserve(bf16 ? bytes * 2 : bytes, bf16 ? gguf::Type::F32 : type, std::move(dims));
    char* dst = static_cast<char*>(d.buf->contents()) + d.off;
    if (bf16) {
        std::vector<uint16_t> raw(bytes / 2);
        gguf::read_exact(fd, raw.data(), bytes, offset);
        auto* f = reinterpret_cast<uint32_t*>(dst);
        for (size_t i = 0; i < raw.size(); ++i)
            f[i] = uint32_t(raw[i]) << 16;
    } else {
        gguf::read_exact(fd, dst, bytes, offset);
    }
    return d;
}

MTL::Buffer* Runtime::scratch(const std::string& name, uint64_t bytes)
{
    auto it = scratch_.find(name);
    if (it == scratch_.end() || it->second->length() < bytes) {
        mtl::Ref<MTL::Buffer> b(ctx_.device()->newBuffer(std::max<uint64_t>(bytes, 16), MTL::ResourceStorageModeShared));
        it = scratch_.insert_or_assign(name, std::move(b)).first;
    }
    return it->second.get();
}

MTL::ComputePipelineState* Runtime::pso(const std::string& fn)
{
    if (repeat_n_ != 1)
        last_pso_ = fn;
    auto it = psos_.find(fn);
    if (it == psos_.end())
        it = psos_.emplace(fn, ctx_.pipeline(lib_.get(), fn)).first;
    return it->second.get();
}

bool Runtime::adjacent(const Tensor& a, const Tensor& b)
{
    if (a.buf != b.buf || a.type != b.type || a.dims != b.dims || a.dims.size() != 2)
        return false;
    return b.off == a.off + mtl::Gemv::row_bytes(a.type, uint32_t(a.dims[0])) * a.dims[1];
}

void Runtime::gemv(MTL::ComputeCommandEncoder*& enc, const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                   uint64_t y_off, uint32_t batch, uint32_t t, bool x_shared)
{
    mtl::Gemv::Args a;
    a.type = w.type;
    a.k = uint32_t(w.dims[0]);
    a.n = uint32_t(w.dims[1]);
    a.t = t;
    a.batch = batch;
    a.x_shared = x_shared;
    const bool labelled = tel_ == Telemetry::kernels || repeat_n_ != 1;
    const std::string label = labelled ? std::format("gemv {} {}x{}", gguf::traits(a.type).name, a.k, a.n) : "";
    for (int r = repeats(label); r > 0; --r)
        gemv_->encode(enc, a, w.buf, w.off, x, x_off, y, y_off);
    if (tel_ == Telemetry::kernels)
        after(enc, label);
}

// y[t] = W x[t] for T tokens: the GEMV for one token, the GEMM for wide matrices, else the GEMV over all tokens.
// y rows are ldy elements apart (0: n).
void Runtime::proj(MTL::ComputeCommandEncoder*& enc, const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y,
                   uint64_t y_off, uint32_t T, uint32_t ldy)
{
    const uint32_t k = uint32_t(w.dims[0]), n = uint32_t(w.dims[1]);
    // A few tokens (decode, speculative verification) stay on the GEMV, which reads each weight once for them.
    if (T >= gemm_min_tokens_ && (n >= 256 || (ldy && ldy != n)) && k % 32 == 0 && mtl::Gemm::supports(w.type)) {
        gemm_->encode(enc, w.type, k, n, T, w.buf, w.off, halves(enc, x, x_off, T, k, 0), 0, y, y_off, ldy);
        if (tel_ == Telemetry::kernels)
            after(enc, std::format("gemm {} {}x{}", gguf::traits(w.type).name, k, n));
        return;
    }
    if (T == 1 || !ldy || ldy == n) {
        gemv(enc, w, x, x_off, y, y_off, 1, T);
        return;
    }
    // Strided rows: one multi-token GEMV (weights read once) into contiguous rows, then copied into place.
    MTL::Buffer* tmp = scratch("proj_rows", uint64_t(T) * n * 4);
    gemv(enc, w, x, x_off, tmp, 0, 1, T);
    const uint32_t a[2] = {n, ldy};
    enc->setComputePipelineState(pso("copy_rows"));
    enc->setBuffer(tmp, 0, 0);
    enc->setBuffer(y, y_off, 1);
    enc->setBytes(a, sizeof(a), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(n, T, 1), MTL::Size(256, 1, 1));
    after(enc, "copy_rows");
}

MTL::Buffer* Runtime::halves(MTL::ComputeCommandEncoder* enc, MTL::Buffer* x, uint64_t x_off, uint32_t rows,
                             uint32_t cols, uint32_t ld)
{
    MTL::Buffer* h = scratch("xh", uint64_t(rows) * cols * 2);
    gemm_->to_half(enc, x, x_off, rows, cols, ld, h, 0);
    return h;
}

void Runtime::dispatch(MTL::ComputeCommandEncoder*& enc, const std::string& fn, std::initializer_list<MTL::Buffer*> bufs,
                       const void* args, size_t args_size, MTL::Size grid, MTL::Size tg, bool threads)
{
    enc->setComputePipelineState(pso(fn));
    NS::UInteger i = 0;
    for (MTL::Buffer* b : bufs)
        enc->setBuffer(b, 0, i++);
    if (args)
        enc->setBytes(args, args_size, i);
    for (int r = repeats(fn); r > 0; --r) {
        if (threads)
            enc->dispatchThreads(grid, tg);
        else
            enc->dispatchThreadgroups(grid, tg);
    }
    after(enc, fn);
}

void Runtime::norm(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, const Tensor* w, MTL::Buffer* y, uint32_t rows,
                   uint32_t n, uint32_t wn, float eps, uint64_t x_off, uint64_t y_off, uint32_t ystride)
{
    const NormArgs a{n, w ? wn : 0, eps, ystride};
    enc->setComputePipelineState(pso("rms_norm"));
    enc->setBuffer(x, x_off, 0);
    enc->setBuffer(w ? w->buf : x, w ? w->off : 0, 1);
    enc->setBuffer(y, y_off, 2);
    enc->setBytes(&a, sizeof(a), 3);
    for (int r = repeats("rms_norm"); r > 0; --r)
        enc->dispatchThreadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "rms_norm");
}

void Runtime::ffn_dense(MTL::ComputeCommandEncoder*& enc, const Tensor& gate, const Tensor& up, const Tensor& down,
                        MTL::Buffer* xn, MTL::Buffer* out, float limit, uint32_t T, uint64_t x_off, uint64_t out_off)
{
    const uint32_t nff = uint32_t(gate.dims[1]);
    const uint64_t rows = uint64_t(T) * nff * 4;
    MTL::Buffer* gu = scratch("ffgu", 2 * rows); // gate rows, then up rows
    MTL::Buffer* m = scratch("ffm", rows);
    if (T < gemm_min_tokens_ && adjacent(gate, up)) {
        gemv(enc, gate, xn, x_off, gu, 0, 2, T, true);
    } else {
        proj(enc, gate, xn, x_off, gu, 0, T);
        proj(enc, up, xn, x_off, gu, rows, T);
    }
    const SwigluArgs sa{T * nff, limit > 1e-6f ? limit : 0.0f};
    enc->setComputePipelineState(pso("swiglu"));
    enc->setBuffer(gu, 0, 0);
    enc->setBuffer(gu, rows, 1);
    enc->setBuffer(m, 0, 2);
    enc->setBytes(&sa, sizeof(sa), 3);
    for (int r = repeats("swiglu"); r > 0; --r)
        enc->dispatchThreads(MTL::Size(T * nff, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "swiglu");
    if (T == 1)
        probe(enc, "imx_ffm", int(cur_layer_), m, nff);
    proj(enc, down, m, 0, out, out_off, T);
}

mtl::Gemv::Job Runtime::gemv_job(const Tensor& w, MTL::Buffer* x, uint64_t x_off, MTL::Buffer* y, uint64_t y_off,
                                 uint32_t batch, uint32_t t, bool x_shared) const
{
    mtl::Gemv::Job j;
    j.a.type = w.type;
    j.a.k = uint32_t(w.dims[0]);
    j.a.n = uint32_t(w.dims[1]);
    j.a.t = t;
    j.a.batch = batch;
    j.a.x_shared = x_shared;
    j.w = w.buf;
    j.w_off = w.off;
    j.x = x;
    j.x_off = x_off;
    j.y = y;
    j.y_off = y_off;
    return j;
}

bool Runtime::gemv_jobs_ok(std::span<const mtl::Gemv::Job> jobs) const
{
    if (jobs.empty() || repeat_n_ != 1 || tel_ == Telemetry::kernels)
        return false;
    for (const auto& j : jobs)
        if (!gemv_->job_supported(j.a) || j.a.t != jobs[0].a.t)
            return false;
    return true;
}

void Runtime::gemv_jobs(MTL::ComputeCommandEncoder*& enc, std::span<const mtl::Gemv::Job> jobs)
{
    constexpr uint64_t kTableBytes = 1 << 20;
    if (!jobs_table_)
        jobs_table_ = mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(kTableBytes, MTL::ResourceStorageModeShared));
    const uint64_t bytes = jobs.size() * mtl::Gemv::kJobBytes;
    if (jobs_off_ + bytes > kTableBytes)
        throw std::runtime_error("too many GEMV jobs in one step");
    gemv_->encode_jobs(enc, jobs, jobs_table_.get(), jobs_off_);
    jobs_off_ += bytes;
}

MTL::ComputeCommandEncoder* Runtime::encoder(MTL::CommandBuffer* cb, uint32_t layer, int section, MTL::DispatchType type)
{
    if (!timing() || (tel_ == Telemetry::layers && section < 0))
        return cb->computeCommandEncoder(type);
    if (tel_ == Telemetry::kernels)
        section = -1;
    MTL::ComputePassDescriptor* d = MTL::ComputePassDescriptor::computePassDescriptor();
    d->setDispatchType(type);
    auto* att = d->sampleBufferAttachments()->object(0);
    if (next_sample_ + 2 > samples_.size() * kSamplesPerBuffer)
        throw std::runtime_error("telemetry: too many timed encoders in one step");
    att->setSampleBuffer(samples_[next_sample_ / kSamplesPerBuffer].get());
    att->setStartOfEncoderSampleIndex(next_sample_ % kSamplesPerBuffer);
    att->setEndOfEncoderSampleIndex(next_sample_ % kSamplesPerBuffer + 1);
    timed_.push_back({layer, section, next_sample_, next_sample_ + 1, {}});
    next_sample_ += 2;
    return cb->computeCommandEncoder(d);
}

void Runtime::after(MTL::ComputeCommandEncoder*& enc, std::string_view kernel)
{
    if (tel_ != Telemetry::kernels || timed_.empty())
        return;
    timed_.back().kernel = kernel;
    timed_.back().layer = cur_layer_;
    enc->endEncoding();
    enc = encoder(cur_cb_, cur_layer_, -1);
}

void Runtime::add_samples(uint64_t want)
{
    NS::Array* sets = ctx_.device()->counterSets();
    MTL::CounterSet* ts = nullptr;
    for (NS::UInteger i = 0; i < sets->count(); ++i)
        if (sets->object<MTL::CounterSet>(i)->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
            ts = sets->object<MTL::CounterSet>(i);
    mtl::Ref<MTL::CounterSampleBufferDescriptor> d(MTL::CounterSampleBufferDescriptor::alloc()->init());
    d->setCounterSet(ts);
    d->setSampleCount(kSamplesPerBuffer);
    d->setStorageMode(MTL::StorageModeShared);
    for (uint64_t n = samples_.size() * kSamplesPerBuffer; n < want; n += kSamplesPerBuffer) {
        NS::Error* err = nullptr;
        samples_.emplace_back(ctx_.device()->newCounterSampleBuffer(d.get(), &err));
        if (!samples_.back())
            throw std::runtime_error(std::format("newCounterSampleBuffer: {}",
                                                 err ? err->localizedDescription()->utf8String() : "?"));
    }
}

void Runtime::enable_part_timing(bool on, uint64_t samples)
{
    part_timing_ = on && tel_ == Telemetry::layers;
    if (part_timing_)
        add_samples(samples);
}

void Runtime::part(MTL::ComputeCommandEncoder*& enc, std::string_view name)
{
    if (!part_timing_ || tel_ != Telemetry::layers || timed_.empty() || timed_.back().section < 0)
        return;
    const Timed cur = timed_.back();
    enc->endEncoding();
    enc = encoder(cur_cb_, cur.layer, cur.section);
    timed_.back().kernel = name;
}

void Runtime::resolve_times(uint64_t cpu0, uint64_t gpu0, const std::function<void(uint32_t, int, double)>& section_ms,
                            std::vector<KernelTime>& parts, std::vector<KernelTime>& kernels)
{
    MTL::Timestamp cpu1 = 0, gpu1 = 0;
    ctx_.device()->sampleTimestamps(&cpu1, &gpu1); // the CPU side is in nanoseconds
    const double ns_per_tick = gpu1 > gpu0 ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;
    std::vector<uint64_t> ts(next_sample_, MTL::CounterErrorValue);
    for (NS::UInteger first = 0; first < next_sample_; first += kSamplesPerBuffer) {
        const NS::UInteger n = std::min<NS::UInteger>(kSamplesPerBuffer, next_sample_ - first);
        NS::Data* data = samples_[first / kSamplesPerBuffer]->resolveCounterRange(NS::Range(0, n));
        if (!data)
            continue;
        const auto* r = static_cast<const MTL::CounterResultTimestamp*>(data->mutableBytes());
        for (NS::UInteger i = 0; i < n; ++i)
            ts[first + i] = r[i].timestamp;
    }
    for (const Timed& t : timed_) {
        const uint64_t a = ts[t.start], b = ts[t.end];
        if (a == MTL::CounterErrorValue || b == MTL::CounterErrorValue || b < a)
            continue;
        const double ms = double(b - a) * ns_per_tick / 1e6;
        if (t.section < 0) {
            if (t.kernel.empty())
                continue;
            auto it = std::ranges::find(kernels, t.kernel, &KernelTime::kernel);
            if (it == kernels.end())
                it = kernels.insert(it, KernelTime{t.kernel, 0, 0});
            it->ms += ms;
            ++it->count;
            continue;
        }
        if (!t.kernel.empty()) {
            auto it = std::ranges::find(parts, t.kernel, &KernelTime::kernel);
            if (it == parts.end())
                it = parts.insert(it, KernelTime{t.kernel, 0, 0});
            it->ms += ms;
            ++it->count;
        }
        section_ms(t.layer, t.section, ms);
    }
    timed_.clear();
    next_sample_ = 0;
}

void Runtime::finish(MTL::CommandBuffer* cb)
{
    auto t0 = Clock::now();
    if (encoding_) {
        stats_.encode_ms += std::chrono::duration<double, std::milli>(t0 - step_start_).count();
        encoding_ = false;
    }
    cb->commit();
    on_committed(cb);
    cb->waitUntilCompleted();
    on_completed();
    stats_.gpu_wait_ms += ms_since(t0);
    ahead_.push_back(cb);
    for (MTL::CommandBuffer* c : ahead_) { // the earlier ones completed first: one queue, in order
        stats_.gpu_exec_ms += (c->GPUEndTime() - c->GPUStartTime()) * 1e3;
        if (c->status() != MTL::CommandBufferStatusCompleted)
            throw std::runtime_error(std::format("command buffer failed: {}",
                                                 c->error() ? c->error()->localizedDescription()->utf8String() : "?"));
    }
    ahead_.clear();
    jobs_off_ = 0; // every dispatch that read the job tables has completed
    deliver_probes();
}

void Runtime::commit_ahead(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc)
{
    if (!commit_ahead_ok())
        return;
    enc->endEncoding();
    if (encoding_) {
        stats_.encode_ms += std::chrono::duration<double, std::milli>(Clock::now() - step_start_).count();
        encoding_ = false;
    }
    cb->commit();
    ahead_.push_back(cb);
    cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    enc = encoder(cb, cur_layer_, -1);
}

void Runtime::submit(MTL::CommandBuffer*& cb, MTL::ComputeCommandEncoder*& enc)
{
    enc->endEncoding();
    finish(cb);
    cb = ctx_.queue()->commandBuffer();
    cur_cb_ = cb;
    enc = encoder(cb, cur_layer_, -1);
}

void Runtime::probe(MTL::ComputeCommandEncoder*& enc, const std::string& name, int layer, MTL::Buffer* buf, size_t n,
                    uint64_t off)
{
    if (!probe_)
        return;
    PendingProbe p{name, layer, mtl::Ref<MTL::Buffer>(ctx_.device()->newBuffer(n * 4, MTL::ResourceStorageModeShared)), n};
    const uint32_t count = uint32_t(n);
    enc->setComputePipelineState(pso("copy_f32"));
    enc->setBuffer(buf, off, 0);
    enc->setBuffer(p.copy.get(), 0, 1);
    enc->setBytes(&count, sizeof(count), 2);
    for (int rep_ = repeats(last_pso_); rep_ > 0; --rep_)
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    after(enc, "probe copy");
    pending_probes_.push_back(std::move(p));
}

void Runtime::deliver_probes()
{
    for (const auto& p : pending_probes_)
        probe_(p.name, p.layer, static_cast<const float*>(p.copy->contents()), p.n);
    pending_probes_.clear();
}

} // namespace mako::gpu
