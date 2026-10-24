// Prefill GEMM: y[t][n] = sum_k W[n][k] * x[t][k] for many tokens, x in half (F32, Q8_0, Q6_K, IQ2_XXS, IQ2_S,
// IQ3_XXS, F16 weights); matmul2d accumulates into a float cooperative tensor (the GPU's matrix units where it has
// them).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Block layouts and the IQ grid/sign tables, from the pinned per-format kernel headers.
#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#include "ggml-common.h"

// attn_pv's tile.
constant int TM = 64, TN = 64, TK = 64;

struct GemmArgs {
    uint t, k, n, row_bytes;
    uint ldy; // y row stride in elements
    uint ldx; // x row stride in elements
};

enum WeightType { F32 = 0, Q8_0 = 1, Q6_K = 2, IQ2_XXS = 3, IQ2_S = 4, IQ3_XXS = 5, F16 = 6 };

// 8 weights from a grid entry (8 or 4 bytes per entry), signs from bit i of sb, times dl.
static void grid8(constant uchar* g, uint sb, float dl, threadgroup half* out)
{
    for (short i = 0; i < 8; ++i)
        out[i] = half(dl * float(g[i]) * ((sb >> i) & 1 ? -1.0f : 1.0f));
}

// Writes 32 weights of `row` starting at k0 (a multiple of 32) to out.
template <int Type>
static void dequant32(device const uchar* row, uint k0, threadgroup half* out)
{
    if (Type == F16) {
        device const half* p = reinterpret_cast<device const half*>(row) + k0;
        for (short i = 0; i < 32; ++i)
            out[i] = p[i];
    } else if (Type == F32) {
        device const float4* p = reinterpret_cast<device const float4*>(row) + k0 / 4;
        for (short i = 0; i < 8; ++i) {
            const float4 v = p[i];
            out[4 * i] = half(v.x);
            out[4 * i + 1] = half(v.y);
            out[4 * i + 2] = half(v.z);
            out[4 * i + 3] = half(v.w);
        }
    } else if (Type == Q8_0) {
        device const uchar* blk = row + (k0 / 32) * 34;
        const float d = float(*reinterpret_cast<device const half*>(blk));
        device const char* q = reinterpret_cast<device const char*>(blk + 2);
        for (short i = 0; i < 32; ++i)
            out[i] = half(d * float(q[i]));
    } else if (Type == IQ2_XXS) {
        // Per 32: four grid indices, then four 7-bit sign indices and a 4-bit scale.
        device const block_iq2_xxs* blk = reinterpret_cast<device const block_iq2_xxs*>(row) + k0 / 256;
        device const ushort* q2 = blk->qs + 4 * ((k0 % 256) / 32);
        const uint idx = uint(q2[0]) | (uint(q2[1]) << 16), aux = uint(q2[2]) | (uint(q2[3]) << 16);
        const float dl = float(blk->d) * (0.5f + float(aux >> 28)) * 0.25f;
        for (short l = 0; l < 4; ++l)
            grid8(reinterpret_cast<constant uchar*>(iq2xxs_grid + ((idx >> (8 * l)) & 255)),
                  ksigns_iq2xs[(aux >> (7 * l)) & 127], dl, out + 8 * l);
    } else if (Type == IQ2_S) {
        // Per 32: four index bytes with two high bits each in qh, four sign bytes, a 4-bit scale per 16.
        device const block_iq2_s* blk = reinterpret_cast<device const block_iq2_s*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32, qh = blk->qh[ib];
        device const uchar* qs = blk->qs + 4 * ib;
        device const uchar* sg = blk->qs + 32 + 4 * ib;
        const float d = float(blk->d);
        for (short l = 0; l < 4; ++l) {
            const float dl = d * (0.5f + float((blk->scales[ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
            grid8(reinterpret_cast<constant uchar*>(iq2s_grid + (qs[l] | (((qh >> (2 * l)) & 3) << 8))), sg[l], dl,
                  out + 8 * l);
        }
    } else if (Type == IQ3_XXS) {
        // Per 32: eight 4-weight grid indices, then four 7-bit sign indices and a 4-bit scale.
        device const block_iq3_xxs* blk = reinterpret_cast<device const block_iq3_xxs*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32;
        device const uchar* q3 = blk->qs + 8 * ib;
        device const ushort* gas = reinterpret_cast<device const ushort*>(blk->qs + 64) + 2 * ib;
        const uint aux = uint(gas[0]) | (uint(gas[1]) << 16);
        const float dl = float(blk->d) * (0.5f + float(aux >> 28)) * 0.5f;
        for (short l = 0; l < 4; ++l) {
            const uint sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            constant uchar* g1 = reinterpret_cast<constant uchar*>(iq3xxs_grid + q3[2 * l]);
            constant uchar* g2 = reinterpret_cast<constant uchar*>(iq3xxs_grid + q3[2 * l + 1]);
            for (short i = 0; i < 4; ++i) {
                out[8 * l + i] = half(dl * float(g1[i]) * ((sb >> i) & 1 ? -1.0f : 1.0f));
                out[8 * l + 4 + i] = half(dl * float(g2[i]) * ((sb >> (i + 4)) & 1 ? -1.0f : 1.0f));
            }
        }
    } else {
        // Q6_K, 210 bytes per 256: ql[128] low nibbles, qh[64] high pairs, scales[16], d. The 32 weights at
        // k0 share half n, quarter qt: low or high nibbles of ql, bit pair qt of qh, scales by 16.
        device const uchar* blk = row + (k0 / 256) * 210;
        const uint j = k0 % 256, n = j / 128, qt = (j % 128) / 32;
        device const uchar* ql = blk + 64 * n + (qt & 1) * 32;
        device const uchar* qh = blk + 128 + 32 * n;
        device const char* sc = reinterpret_cast<device const char*>(blk + 192) + 8 * n + 2 * qt;
        const float d = float(*reinterpret_cast<device const half*>(blk + 208));
        const float d0 = d * float(sc[0]), d1 = d * float(sc[1]);
        for (short l = 0; l < 32; ++l) {
            const int lo = qt < 2 ? (ql[l] & 0xF) : (ql[l] >> 4);
            const int q = (lo | (((qh[l] >> (2 * qt)) & 3) << 4)) - 32;
            out[l] = half((l < 16 ? d0 : d1) * float(q));
        }
    }
}

// Threadgroups of 4 simdgroups for TM tokens x TN rows; grid (ceil(n / TN), ceil(t / TM)). The half activations
// go to matmul2d straight from device memory; the weights are dequantized 256 columns at a time into threadgroup
// memory (fewer, longer matmul2d steps run much faster than 64-wide ones). k must be a multiple of 32.
template <int Type, int TM, int TN>
kernel void gemm(device const uchar* w [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]],
                 constant GemmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                 ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int KD = 256, NSG = 4;
    threadgroup half ws[TN * KD];
    const uint n0 = tg.x * TN, t0 = tg.y * TM;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tx(const_cast<device half*>(x), dextents<int32_t, 2>(a.k, a.t),
                                                                array<int32_t, 2>{1, int32_t(a.ldx)});
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw(ws, dextents<int32_t, 2>(KD, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, KD, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mx = tx.slice(0, int(t0));
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(tw), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    for (uint k = 0; k < a.k; k += KD) {
        // Each 32-wide piece of a weight row is staged by one thread; past k, zeros.
        for (uint s = tid; s < TN * (KD / 32); s += NSG * 32) {
            const uint r = s / (KD / 32), c = (s % (KD / 32)) * 32;
            threadgroup half* wo = ws + r * KD + c;
            if (n0 + r < a.n && k + c < a.k) {
                dequant32<Type>(w + uint64_t(n0 + r) * a.row_bytes, k + c, wo);
            } else {
                for (short i = 0; i < 32; ++i)
                    wo[i] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xk = tx.slice(int(k), int(t0));
        op.run(xk, tw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> ty(y, dextents<int32_t, 2>(a.n, a.t),
                                                                 array<int32_t, 2>{1, int32_t(a.ldy)});
    acc.store(ty.slice(n0, t0));
}

template [[host_name("gemm_f32_64x32")]] kernel void gemm<F32, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f32_32x32")]] kernel void gemm<F32, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0_64x32")]] kernel void gemm<Q8_0, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0_32x32")]] kernel void gemm<Q8_0, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k_64x32")]] kernel void gemm<Q6_K, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k_32x32")]] kernel void gemm<Q6_K, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xxs_64x32")]] kernel void gemm<IQ2_XXS, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xxs_32x32")]] kernel void gemm<IQ2_XXS, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s_64x32")]] kernel void gemm<IQ2_S, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s_32x32")]] kernel void gemm<IQ2_S, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs_64x32")]] kernel void gemm<IQ3_XXS, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs_32x32")]] kernel void gemm<IQ3_XXS, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f16_64x32")]] kernel void gemm<F16, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f16_32x32")]] kernel void gemm<F16, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);


// y[r][c] = half(x[r * ldx + c]), dense rows of `cols`; grid (cols, rows).
struct HalfArgs {
    uint cols, ldx;
};
kernel void to_half(device const float* x [[buffer(0)]], device half* y [[buffer(1)]], constant HalfArgs& a [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.cols)
        y[uint64_t(gid.y) * a.cols + gid.x] = half(x[uint64_t(gid.y) * a.ldx + gid.x]);
}

struct PvArgs {
    uint heads, dim, n_kv;
    uint keys_per_split; // a multiple of 64
};

// Attention output for one query position over a shared latent cache, split over keys:
// partial[split][h][d] = sum over the split's keys t of p[h][t] * kv[t][d]. Threadgroups (dim / 64,
// ceil(heads / 64), splits) of 4 simdgroups; keys past n_kv count as zero.
kernel void attn_pv(device const float* p [[buffer(0)]], device const half* kv [[buffer(1)]],
                    device float* partial [[buffer(2)]], constant PvArgs& a [[buffer(3)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup half ps[TM * TK], ks[TK * TN];
    const uint d0 = tg.x * TN, h0 = tg.y * TM;
    const uint t_begin = tg.z * a.keys_per_split, t_end = min(a.n_kv, t_begin + a.keys_per_split);
    const uint r = tid / 2, c = (tid % 2) * 32;

    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(ps, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tk(ks, dextents<int32_t, 2>(TN, TK));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;

    for (uint t0 = t_begin; t0 < t_end; t0 += TK) {
        // p tile: head h0 + r, keys t0 + c ..; kv tile: key t0 + r, dims d0 + c ..
        threadgroup half* po = ps + r * TK + c;
        threadgroup half* ko = ks + r * TN + c;
        for (short i = 0; i < 32; ++i) {
            const uint t = t0 + c + i;
            po[i] = (h0 + r < a.heads && t < t_end) ? half(p[uint64_t(h0 + r) * a.n_kv + t]) : 0.0h;
        }
        const uint tk_row = t0 + r;
        device const half* kr = kv + uint64_t(tk_row) * a.dim + d0 + c;
        for (short i = 0; i < 32; ++i)
            ko[i] = tk_row < t_end ? kr[i] : 0.0h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tp, tk, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(partial + uint64_t(tg.z) * a.heads * a.dim,
                                                                 dextents<int32_t, 2>(a.dim, a.heads));
    acc.store(to.slice(d0, h0));
}

// Sparse attention over a shared half latent cache: each query token attends to the cache rows in its own index list
// (list rows of `stride` entries, count[t] used), so the work per token is bounded by the list, not the context.
struct GatherArgs {
    uint heads;  // query rows per token
    uint dim;    // latent width (multiple of 256)
    uint stride; // list and score row length
    uint splits; // attn_gather_pv: key ranges per token
    uint tokens; // attn_gather_pv: partial is [splits][tokens][heads][dim]
};

// scores[t][h][s] = q[t][h] . kv[list[t][s]]; grid (ceil(stride / 32), 1, tokens), 4 simdgroups; heads <= 64.
kernel void attn_gather_scores(device const half* q [[buffer(0)]], device const half* kv [[buffer(1)]],
                               device const uint* list [[buffer(2)]], device const uint* count [[buffer(3)]],
                               device float* scores [[buffer(4)]], constant GatherArgs& a [[buffer(5)]],
                               uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int GM = 64, GN = 32, GK = 256;
    threadgroup half ws[GN * GK];
    const uint t = tg.z, n0 = tg.x * GN, cnt = count[t];
    list += uint64_t(t) * a.stride;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tx(const_cast<device half*>(q) + uint64_t(t) * a.heads * a.dim,
                                                                dextents<int32_t, 2>(a.dim, a.heads));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw(ws, dextents<int32_t, 2>(GK, GN));
    constexpr auto desc = matmul2d_descriptor(GM, GN, GK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(tw), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    for (uint k = 0; k < a.dim; k += GK) {
        for (uint s = tid; s < GN * (GK / 32); s += 128) {
            const uint r = s / (GK / 32), c = (s % (GK / 32)) * 32;
            threadgroup half* wo = ws + r * GK + c;
            if (n0 + r < cnt) {
                device const half* src = kv + uint64_t(list[n0 + r]) * a.dim + k + c;
                for (short i = 0; i < 32; ++i)
                    wo[i] = src[i];
            } else {
                for (short i = 0; i < 32; ++i)
                    wo[i] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto xk = tx.slice(int(k), 0);
        op.run(xk, tw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> ty(scores + uint64_t(t) * a.heads * a.stride,
                                                                 dextents<int32_t, 2>(a.stride, a.heads));
    acc.store(ty.slice(n0, 0));
}

// partial[split][t][h][d] = sum over the split's list entries s of p[t][h][s] * kv[list[t][s]][d]; grid (dim / 128,
// 1, tokens * splits), 8 simdgroups; heads <= 64, entries past count[t] count as zero. Wide output tiles so each
// token's probabilities are read dim / 128 times.
kernel void attn_gather_pv(device const float* p [[buffer(0)]], device const half* kv [[buffer(1)]],
                           device const uint* list [[buffer(2)]], device const uint* count [[buffer(3)]],
                           device float* partial [[buffer(4)]], constant GatherArgs& a [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]])
{
    constexpr int PN = 128;
    threadgroup half ps[TM * TK], ks[TK * PN];
    const uint t = tg.z / a.splits, split = tg.z % a.splits, d0 = tg.x * PN, cnt = count[t];
    const uint per = (a.stride + a.splits * TK - 1) / (a.splits * TK) * TK;
    const uint s_begin = split * per, s_end = min(cnt, s_begin + per);
    // Loads: thread tid fills 16 probabilities of head r and 32 latent values of entry r.
    const uint r = tid / 4, pc = (tid % 4) * 16, kc = (tid % 4) * 32;
    p += uint64_t(t) * a.heads * a.stride;
    list += uint64_t(t) * a.stride;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(ps, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tk(ks, dextents<int32_t, 2>(PN, TK));
    constexpr auto desc = matmul2d_descriptor(TM, PN, TK, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<8>> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    for (uint s0 = s_begin; s0 < s_end; s0 += TK) {
        threadgroup half* po = ps + r * TK + pc;
        const uint sp = s0 + pc;
        if (r < a.heads && sp + 16 <= s_end) {
            device const float4* pr = reinterpret_cast<device const float4*>(p + uint64_t(r) * a.stride + sp);
            for (short i = 0; i < 4; ++i) {
                const half4 v = half4(pr[i]);
                for (short j = 0; j < 4; ++j)
                    po[i * 4 + j] = v[j];
            }
        } else {
            for (short i = 0; i < 16; ++i)
                po[i] = (r < a.heads && sp + i < s_end) ? half(p[uint64_t(r) * a.stride + sp + i]) : 0.0h;
        }
        threadgroup half4* ko = reinterpret_cast<threadgroup half4*>(ks + r * PN + kc);
        const uint sr = s0 + r;
        if (sr < s_end) {
            device const half4* kr = reinterpret_cast<device const half4*>(kv + uint64_t(list[sr]) * a.dim + d0 + kc);
            for (short i = 0; i < 8; ++i)
                ko[i] = kr[i];
        } else {
            for (short i = 0; i < 8; ++i)
                ko[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tp, tk, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(
        partial + (uint64_t(split) * a.tokens + t) * a.heads * a.dim, dextents<int32_t, 2>(a.dim, a.heads));
    acc.store(to.slice(d0, 0));
}

// Indexer dot products for consecutive tokens: dots[t][h][p] = q[t][h] . pooled[p] in float (matmul2d float x float),
// over the pools token 0 can see and beyond (the reduction stops at each token's own count). Grid (ceil(pools / 64),
// 1, tokens), 4 simdgroups; heads <= 32 (padded to 32 rows), dim the reduction length.
struct IdxDotArgs {
    uint heads, dim, pools; // pools: of the last token
    uint stride;            // dots row length
};
kernel void idx_dots(device const float* q [[buffer(0)]], device const float* pooled [[buffer(1)]],
                     device float* dots [[buffer(2)]], constant IdxDotArgs& a [[buffer(3)]],
                     uint3 tg [[threadgroup_position_in_grid]])
{
    const uint t = tg.z, p0 = tg.x * 64;
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device float*>(q) + uint64_t(t) * a.heads * a.dim,
                                                                 dextents<int32_t, 2>(a.dim, a.heads));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device float*>(pooled),
                                                                 dextents<int32_t, 2>(a.dim, a.pools));
    constexpr auto desc = matmul2d_descriptor(32, 64, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    tensor<device float, dextents<int32_t, 2>, tensor_inline> td(dots + uint64_t(t) * a.heads * a.stride,
                                                                 dextents<int32_t, 2>(a.pools, a.heads),
                                                                 array<int32_t, 2>{1, int32_t(a.stride)});
    auto mq = tq.slice(0, 0);
    auto mk = tk.slice(0, int(p0));
    auto md = td.slice(int(p0), 0);
    op.run(mq, mk, md);
}
