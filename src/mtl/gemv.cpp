#include "mtl/gemv.hpp"

#include "mtl/embedded.hpp"

#include <format>
#include <string_view>
#include <stdexcept>

namespace mako::mtl {

namespace {

// Layout must match the kernels' argument struct.
struct KArgs {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int32_t nr0;
    int16_t r2, r3;
};

// Layout must match the indexed kernels' argument struct.
struct KArgsId {
    int32_t nei0, nei1;
    uint64_t nbi1;
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1;
    uint64_t nb1;
    int32_t nr0;
};

struct TypeInfo {
    const char* kernel;
    int nsg, nr0, smem;
    bool rows_per_sg_grid;
};

const TypeInfo* info(gguf::Type t)
{
    using gguf::Type;
    static const TypeInfo q8_0{"kernel_mul_mv_q8_0_f32", 4, 2, 32 * 4 * 2, true};
    static const TypeInfo q6_k{"kernel_mul_mv_q6_K_f32", 2, 2, 0, false};
    static const TypeInfo q5_k{"kernel_mul_mv_q5_K_f32", 2, 1, 0, false};
    static const TypeInfo q4_k{"kernel_mul_mv_q4_K_f32", 2, 2, 0, false};
    static const TypeInfo q3_k{"kernel_mul_mv_q3_K_f32", 2, 2, 0, false};
    static const TypeInfo q2_k{"kernel_mul_mv_q2_K_f32", 2, 4, 0, false};
    static const TypeInfo q4_0{"kernel_mul_mv_q4_0_f32", 2, 4, 0, false};
    static const TypeInfo iq4_xs{"kernel_mul_mv_iq4_xs_f32", 2, 2, 32 * 4, false};
    static const TypeInfo iq3_s{"kernel_mul_mv_iq3_s_f32", 2, 4, 512 * 4, false};
    static const TypeInfo iq3_xxs{"kernel_mul_mv_iq3_xxs_f32", 2, 4, 256 * 4 + 128, false};
    static const TypeInfo iq2_s{"kernel_mul_mv_iq2_s_f32", 2, 4, 0, false};
    static const TypeInfo iq2_xs{"kernel_mul_mv_iq2_xs_f32", 2, 4, 512 * 8 + 128, false};
    static const TypeInfo iq2_xxs{"kernel_mul_mv_iq2_xxs_f32", 2, 4, 256 * 8 + 128, false};
    static const TypeInfo iq4_nl{"kernel_mul_mv_iq4_nl_f32", 2, 2, 32 * 4, false};
    static const TypeInfo iq1_s{"kernel_mul_mv_iq1_s_f32", 2, 4, 0, false};
    static const TypeInfo iq1_m{"kernel_mul_mv_iq1_m_f32", 2, 4, 0, false};
    static const TypeInfo mxfp4{"kernel_mul_mv_mxfp4_f32", 2, 2, 32 * 4, false};
    switch (t) {
    case Type::Q8_0: return &q8_0;
    case Type::Q6_K: return &q6_k;
    case Type::Q5_K: return &q5_k;
    case Type::Q4_K: return &q4_k;
    case Type::Q3_K: return &q3_k;
    case Type::Q2_K: return &q2_k;
    case Type::Q4_0: return &q4_0;
    case Type::IQ4_XS: return &iq4_xs;
    case Type::IQ3_S: return &iq3_s;
    case Type::IQ3_XXS: return &iq3_xxs;
    case Type::IQ2_S: return &iq2_s;
    case Type::IQ2_XS: return &iq2_xs;
    case Type::IQ2_XXS: return &iq2_xxs;
    case Type::IQ4_NL: return &iq4_nl;
    case Type::IQ1_S: return &iq1_s;
    case Type::IQ1_M: return &iq1_m;
    case Type::MXFP4: return &mxfp4;
    default: return nullptr;
    }
}

bool is_iq(gguf::Type t)
{
    using gguf::Type;
    return t == Type::IQ3_S || t == Type::IQ3_XXS || t == Type::IQ2_S || t == Type::IQ2_XS || t == Type::IQ2_XXS ||
           t == Type::IQ1_S || t == Type::IQ1_M;
}

// Layout must match ShortArgs in gemv.metal.
struct ShortArgs {
    uint32_t k, n, row_bytes, tokens;
};

// Layout must match GemvJob in gemv_slots.metal (pointers as GPU addresses).
struct JobRow {
    uint64_t w, x, y, ids, w_tail;
    uint64_t nb02, nb01, x_tok, x_mat, y_tok, y_mat;
    uint32_t ids_stride, split, type, k, n, nr0, tg0, tgs_x, nb00;
};
static_assert(sizeof(JobRow) == 128);

// The job kernel's type code, rows per simdgroup and rows per threadgroup (4 simdgroups), or -1.
int job_type(gguf::Type t, int& nr0, int& rows_per_tg)
{
    using gguf::Type;
    switch (t) {
    case Type::Q6_K: nr0 = 2, rows_per_tg = 8; return 0;
    case Type::Q8_0: nr0 = 2, rows_per_tg = 2; return 1;  // the simdgroups split K
    case Type::F32: nr0 = 2, rows_per_tg = 2; return 2;   // the same
    case Type::IQ2_XXS: nr0 = 4, rows_per_tg = 16; return 3;
    case Type::IQ3_XXS: nr0 = 4, rows_per_tg = 16; return 4;
    case Type::IQ2_S: nr0 = 4, rows_per_tg = 16; return 5;
    case Type::IQ3_S: nr0 = 4, rows_per_tg = 16; return 6;
    default: return -1;
    }
}

} // namespace

Gemv::Gemv(const Context& ctx) : ctx_(ctx), lib_(ctx.load(gemv_metallib())), own_lib_(ctx.load(kernels_metallib())) {}

MTL::ComputePipelineState* Gemv::own(const std::string& name)
{
    auto it = own_psos_.find(name);
    if (it == own_psos_.end())
        it = own_psos_.emplace(name, ctx_.pipeline(own_lib_.get(), name)).first;
    return it->second.get();
}

bool Gemv::encode_short(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                        const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off)
{
    const bool q8 = a.type == gguf::Type::Q8_0;
    if (!own_kernels_ || (!q8 && a.type != gguf::Type::F32) || a.batch != 1 || a.t > 4 || a.k % 32 || a.k > 512)
        return false;
    const ShortArgs sa{a.k, a.n, uint32_t(row_bytes(a.type, a.k)), a.t};
    enc->setComputePipelineState(own(q8 ? "gemv_short_q8_0" : "gemv_short_f32"));
    enc->setBuffer(w, w_off, 0);
    enc->setBuffer(x, x_off, 1);
    enc->setBuffer(y, y_off, 2);
    enc->setBytes(&sa, sizeof(sa), 3);
    enc->dispatchThreadgroups(MTL::Size((a.n + 63) / 64, 1, 1), MTL::Size(256, 1, 1));
    return true;
}

bool Gemv::supports(gguf::Type type) { return type == gguf::Type::F32 || info(type) != nullptr; }

uint64_t Gemv::row_bytes(gguf::Type type, uint32_t k)
{
    const auto tr = gguf::traits(type);
    return uint64_t(k) / tr.block_elems * tr.block_bytes;
}

Gemv::Kernel& Gemv::kernel(gguf::Type type, uint32_t k, uint32_t batch, bool indexed, uint32_t tokens)
{
    const bool vec4 = type == gguf::Type::F32 && k % 4 == 0;
    if (indexed)
        batch = 1;
    auto key = std::make_tuple(type, vec4, batch, indexed, tokens);
    if (auto it = kernels_.find(key); it != kernels_.end())
        return it->second;

    Kernel kern;
    std::string name;
    if (type == gguf::Type::F32) {
        name = vec4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32";
        kern.nsg = std::min<int>(4, (int(k) + 127) / 128);
        kern.nr0 = 2;
        kern.smem = 32 * 4 * kern.nr0;
        kern.rows_per_sg_grid = true;
    } else {
        const TypeInfo* ti = info(type);
        if (!ti)
            throw std::runtime_error(std::format("no GEMV kernel for {}", gguf::traits(type).name));
        name = ti->kernel;
        kern.nsg = ti->nsg;
        kern.nr0 = ti->nr0;
        kern.smem = ti->smem;
        kern.rows_per_sg_grid = ti->rows_per_sg_grid;
    }
    // mako's IQ2_XXS / IQ3_XXS kernels, faster (grid tables as half): IQ2_XXS gives the per-format kernel's results bit
    // for bit, IQ3_XXS rounds differently (~1e-7 relative).
    // mako's IQ1_S / IQ1_M kernels: the same sums, vectorized.
    const bool own_iq = own_kernels_ && tokens == 1 &&
                        (type == gguf::Type::IQ2_XXS || type == gguf::Type::IQ3_XXS || type == gguf::Type::IQ1_S ||
                         type == gguf::Type::IQ1_M);
    if (own_iq) {
        kern.nr0 = 4;
        switch (type) {
        case gguf::Type::IQ2_XXS: name = "mako_mul_mv_iq2_xxs_f32_r4", kern.smem = 256 * 16 + 128; break;
        case gguf::Type::IQ3_XXS: name = "mako_mul_mv_iq3_xxs_f32_r4", kern.smem = 256 * 8 + 128; break;
        case gguf::Type::IQ1_S: name = "mako_mul_mv_iq1_s_f32_r4", kern.smem = 0; break;
        default: name = "mako_mul_mv_iq1_m_f32_r4", kern.smem = 0; break;
        }
    }

    if (indexed && own_iq)
        name.replace(0, std::string_view("mako_mul_mv_").size(), "mako_mul_mv_slots_mako_");
    else if (indexed)
        name.replace(0, std::string_view("kernel_mul_mv_").size(), "mako_mul_mv_slots_");
    else if (tokens > 1)
        name = std::format("mako_mul_mv_q6_K_t{}", tokens);

    Ref<MTL::FunctionConstantValues> fc(MTL::FunctionConstantValues::alloc()->init());
    const int16_t nsg = int16_t(kern.nsg), nxpsg = 16, ne12 = int16_t(batch), one = 1;
    const bool split = false;
    fc->setConstantValue(&nsg, MTL::DataTypeShort, 600);
    fc->setConstantValue(&nxpsg, MTL::DataTypeShort, 601);
    fc->setConstantValue(&ne12, MTL::DataTypeShort, 602);
    fc->setConstantValue(&one, MTL::DataTypeShort, 603);
    fc->setConstantValue(&one, MTL::DataTypeShort, 604);
    fc->setConstantValue(&split, MTL::DataTypeBool, 605);
    kern.pso = ctx_.pipeline(lib_.get(), name, fc.get());
    return kernels_.emplace(key, std::move(kern)).first->second;
}

void Gemv::encode(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                  const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off)
{
    if (encode_short(enc, a, w, w_off, x, x_off, y, y_off))
        return;
    if (is_iq(a.type) && a.k < 1024)
        throw std::runtime_error(std::format("{} GEMV with k={} needs the split-row variant", gguf::traits(a.type).name, a.k));
    if (a.batch > 32767)
        throw std::runtime_error("GEMV batch too large");
    // Q6_K for 2 tokens: one threadgroup row for both, each weight block decoded once.
    const uint32_t tokens = own_kernels_ && a.type == gguf::Type::Q6_K && a.t == 2 ? 2 : 1; // 4: slower (spills)
    Kernel& kern = kernel(a.type, a.k, a.batch, false, tokens);

    const uint64_t rb = row_bytes(a.type, a.k);
    KArgs ka{};
    ka.ne00 = int32_t(a.k);
    ka.ne01 = int32_t(a.n);
    ka.ne02 = int32_t(a.batch);
    ka.nb00 = gguf::traits(a.type).block_bytes;
    ka.nb01 = rb;
    ka.nb02 = rb * a.n;
    ka.nb03 = ka.nb02 * a.batch;
    ka.ne10 = int32_t(a.k);
    ka.ne11 = int32_t(a.t);
    ka.ne12 = int32_t(a.batch);
    ka.nb10 = 4;
    ka.nb11 = uint64_t(a.k) * 4;
    ka.nb12 = a.x_shared ? 0 : ka.nb11 * a.t;
    ka.nb13 = ka.nb12 * a.batch;
    ka.ne0 = int32_t(a.n);
    ka.ne1 = int32_t(a.t);
    ka.nr0 = kern.nr0;
    ka.r2 = 1;
    ka.r3 = 1;

    enc->setComputePipelineState(kern.pso.get());
    enc->setBytes(&ka, sizeof(ka), 0);
    enc->setBuffer(w, w_off, 1);
    enc->setBuffer(x, x_off, 2);
    enc->setBuffer(y, y_off, 3);
    if (kern.smem)
        enc->setThreadgroupMemoryLength(kern.smem, 0);
    const int rows_per_tg = kern.rows_per_sg_grid ? kern.nr0 : kern.nr0 * kern.nsg;
    enc->dispatchThreadgroups(MTL::Size((a.n + rows_per_tg - 1) / rows_per_tg, tokens > 1 ? 1 : a.t, a.batch),
                              MTL::Size(32, kern.nsg, 1));
}

bool Gemv::encode_router(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                         const MTL::Buffer* x, uint64_t x_off, const MTL::Buffer* y, uint64_t y_off,
                         const std::function<void(MTL::ComputeCommandEncoder*)>& bind_route)
{
    if (!own_kernels_ || a.type != gguf::Type::F32 || a.k % 4 || a.batch != 1 || a.n > 512 || a.t > 8)
        return false;
    if (!router_pso_) {
        Ref<MTL::FunctionConstantValues> fc(MTL::FunctionConstantValues::alloc()->init());
        const int16_t nsg = int16_t(std::min<int>(4, (int(a.k) + 127) / 128)), nxpsg = 16, one = 1;
        const bool split = false;
        fc->setConstantValue(&nsg, MTL::DataTypeShort, 600);
        fc->setConstantValue(&nxpsg, MTL::DataTypeShort, 601);
        fc->setConstantValue(&one, MTL::DataTypeShort, 602);
        fc->setConstantValue(&one, MTL::DataTypeShort, 603);
        fc->setConstantValue(&one, MTL::DataTypeShort, 604);
        fc->setConstantValue(&split, MTL::DataTypeBool, 605);
        router_pso_ = ctx_.pipeline(lib_.get(), "mako_mul_mv_f32_f32_4_route", fc.get());
        router_nsg_ = nsg;
    }
    if (router_nsg_ != std::min<int>(4, (int(a.k) + 127) / 128))
        return false; // one k per process is all the engines need
    const uint64_t rb = uint64_t(a.k) * 4;
    KArgs ka{};
    ka.ne00 = int32_t(a.k);
    ka.ne01 = int32_t(a.n);
    ka.ne02 = 1;
    ka.nb00 = 4;
    ka.nb01 = rb;
    ka.nb02 = rb * a.n;
    ka.nb03 = ka.nb02;
    ka.ne10 = int32_t(a.k);
    ka.ne11 = int32_t(a.t);
    ka.ne12 = 1;
    ka.nb10 = 4;
    ka.nb11 = rb;
    ka.nb12 = rb * a.t;
    ka.nb13 = ka.nb12;
    ka.ne0 = int32_t(a.n);
    ka.ne1 = int32_t(a.t);
    ka.nr0 = 2;
    ka.r2 = 1;
    ka.r3 = 1;
    enc->setComputePipelineState(router_pso_.get());
    enc->setBytes(&ka, sizeof(ka), 0);
    enc->setBuffer(w, w_off, 1);
    enc->setBuffer(x, x_off, 2);
    enc->setBuffer(y, y_off, 3);
    bind_route(enc);
    enc->setThreadgroupMemoryLength(32 * 4 * 2, 0);
    enc->dispatchThreadgroups(MTL::Size((a.n + 1) / 2, a.t, 1), MTL::Size(32, router_nsg_, 1));
    return true;
}

bool Gemv::job_supported(const Args& a) const
{
    int nr0, rows;
    if (!own_kernels_ || a.t > 8 || job_type(a.type, nr0, rows) < 0)
        return false;
    if (a.type == gguf::Type::F32 || a.type == gguf::Type::Q8_0)
        return a.k % 32 == 0 && a.k >= 512 && a.k % 4 == 0;
    return a.k >= 1024 && a.k % 256 == 0;
}

void Gemv::encode_jobs(MTL::ComputeCommandEncoder* enc, std::span<const Job> jobs, MTL::Buffer* table,
                       uint64_t table_off)
{
    if (jobs.empty())
        return;
    if (!jobs_pso_) {
        Ref<MTL::FunctionConstantValues> fc(MTL::FunctionConstantValues::alloc()->init());
        const int16_t nsg = 4, nxpsg = 16, one = 1;
        const bool split = false;
        fc->setConstantValue(&nsg, MTL::DataTypeShort, 600);
        fc->setConstantValue(&nxpsg, MTL::DataTypeShort, 601);
        fc->setConstantValue(&one, MTL::DataTypeShort, 602);
        fc->setConstantValue(&one, MTL::DataTypeShort, 603);
        fc->setConstantValue(&one, MTL::DataTypeShort, 604);
        fc->setConstantValue(&split, MTL::DataTypeBool, 605);
        jobs_pso_ = ctx_.pipeline(lib_.get(), "mako_mul_mv_jobs", fc.get());
    }
    auto* rows = reinterpret_cast<JobRow*>(static_cast<char*>(table->contents()) + table_off);
    const uint32_t T = jobs[0].a.t;
    uint32_t tg = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Job& j = jobs[i];
        const Args& a = j.a;
        int nr0 = 0, rows_per_tg = 0;
        const int type = job_type(a.type, nr0, rows_per_tg);
        if (type < 0 || a.t != T || !job_supported(a))
            throw std::runtime_error("GEMV job not supported");
        const uint64_t rb = a.type == gguf::Type::F32 ? uint64_t(a.k) * 4 : row_bytes(a.type, a.k);
        JobRow r{};
        r.w = j.w->gpuAddress() + j.w_off;
        r.x = j.x->gpuAddress() + j.x_off;
        r.y = j.y->gpuAddress() + j.y_off;
        r.nb01 = rb;
        r.nb00 = a.type == gguf::Type::F32 ? 4 : gguf::traits(a.type).block_bytes;
        r.type = uint32_t(type);
        r.k = a.k;
        r.n = a.n;
        r.nr0 = uint32_t(nr0);
        enc->useResource(j.w, MTL::ResourceUsageRead);
        enc->useResource(j.x, MTL::ResourceUsageRead);
        enc->useResource(j.y, MTL::ResourceUsageWrite);
        if (j.ids) {
            r.ids = j.ids->gpuAddress() + j.ids_off;
            r.ids_stride = a.ids_stride ? a.ids_stride / 4 : a.batch;
            r.nb02 = j.stride;
            r.w_tail = j.tail.buf ? j.tail.buf->gpuAddress() + j.tail.off : r.w;
            r.split = j.tail.split;
            const uint64_t nx = a.x_shared ? 1 : a.batch;
            r.x_tok = uint64_t(a.k) * 4 * nx;
            r.x_mat = a.x_shared ? 0 : uint64_t(a.k) * 4;
            r.y_tok = uint64_t(a.n) * 4 * a.batch;
            r.y_mat = uint64_t(a.n) * 4;
            enc->useResource(j.ids, MTL::ResourceUsageRead);
            if (j.tail.buf)
                enc->useResource(j.tail.buf, MTL::ResourceUsageRead);
        } else {
            r.nb02 = rb * a.n;
            r.x_tok = uint64_t(a.k) * 4;
            r.x_mat = a.x_shared ? 0 : uint64_t(a.k) * 4 * T;
            r.y_tok = uint64_t(j.ldy ? j.ldy : a.n) * 4;
            r.y_mat = r.y_tok * T;
        }
        r.tgs_x = (a.n + rows_per_tg - 1) / rows_per_tg;
        r.tg0 = tg;
        tg += r.tgs_x * a.batch;
        rows[i] = r;
    }
    const uint32_t n = uint32_t(jobs.size());
    enc->setComputePipelineState(jobs_pso_.get());
    enc->setBuffer(table, table_off, 0);
    enc->setBytes(&n, sizeof(n), 1);
    enc->setThreadgroupMemoryLength(256 * 16 + 128, 0);
    enc->dispatchThreadgroups(MTL::Size(tg, T, 1), MTL::Size(32, 4, 1));
}

void Gemv::encode_indexed(MTL::ComputeCommandEncoder* enc, const Args& a, const MTL::Buffer* w, uint64_t w_off,
                          uint64_t stride, const MTL::Buffer* ids, uint64_t ids_off, const MTL::Buffer* x,
                          uint64_t x_off, const MTL::Buffer* y, uint64_t y_off, const MTL::Buffer* indirect,
                          uint64_t indirect_off, const Tail& tail)
{
    if (is_iq(a.type) && a.k < 1024)
        throw std::runtime_error(std::format("{} GEMV with k={} needs the split-row variant", gguf::traits(a.type).name, a.k));
    Kernel& kern = kernel(a.type, a.k, 1, true);

    // a.t tokens, each with a row of a.batch ids (ids_stride bytes apart) and its own x row(s).
    const int32_t nx = a.x_shared ? 1 : int32_t(a.batch);
    KArgsId ka{};
    ka.nei0 = int32_t(a.batch);
    ka.nei1 = int32_t(a.t);
    ka.nbi1 = a.ids_stride ? a.ids_stride : uint64_t(a.batch) * 4;
    ka.ne00 = int32_t(a.k);
    ka.ne01 = int32_t(a.n);
    ka.ne02 = 1;
    ka.nb00 = gguf::traits(a.type).block_bytes;
    ka.nb01 = row_bytes(a.type, a.k);
    ka.nb02 = stride;
    ka.ne10 = int32_t(a.k);
    ka.ne11 = nx;
    ka.ne12 = 1;
    ka.ne13 = 1;
    ka.nb10 = 4;
    ka.nb11 = uint64_t(a.k) * 4;
    ka.nb12 = ka.nb11 * uint64_t(nx);
    ka.ne0 = int32_t(a.n);
    ka.ne1 = int32_t(a.batch);
    ka.nb1 = uint64_t(a.n) * 4;
    ka.nr0 = kern.nr0;

    enc->setComputePipelineState(kern.pso.get());
    enc->setBytes(&ka, sizeof(ka), 0);
    enc->setBuffer(w, w_off, 1);
    enc->setBuffer(x, x_off, 2);
    enc->setBuffer(y, y_off, 3);
    enc->setBuffer(ids, ids_off, 4);
    enc->setBuffer(tail.buf ? tail.buf : w, tail.buf ? tail.off : w_off, 5);
    enc->setBytes(&tail.split, sizeof(tail.split), 6);
    if (kern.smem)
        enc->setThreadgroupMemoryLength(kern.smem, 0);
    if (indirect)
        enc->dispatchThreadgroups(indirect, indirect_off, MTL::Size(32, kern.nsg, 1));
    else
        enc->dispatchThreadgroups(indexed_grid(a), MTL::Size(32, kern.nsg, 1));
}

MTL::Size Gemv::indexed_grid(const Args& a)
{
    const Kernel& kern = kernel(a.type, a.k, 1, true);
    const int rows_per_tg = kern.rows_per_sg_grid ? kern.nr0 : kern.nr0 * kern.nsg;
    return MTL::Size((a.n + rows_per_tg - 1) / rows_per_tg, 1, uint64_t(a.batch) * a.t);
}

} // namespace mako::mtl
