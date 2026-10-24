// Prefill GEMM: y[t][n] = sum_k W[n][k] * x[t][k] for many tokens (F32, Q8_0, Q6_K, IQ2_XXS, IQ2_S, IQ3_XXS). Each threadgroup computes 64 tokens x 64 rows:
// per 64-wide k step it dequantizes the weight tile and converts the activation tile to half in threadgroup
// memory, and matmul2d accumulates into a float cooperative tensor (the GPU's matrix units where it has them).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Block layouts and the IQ grid/sign tables, from the pinned per-format kernel headers.
#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#include "ggml-common.h"

constant int TM = 64, TN = 64, TK = 64;

struct GemmArgs {
    uint t, k, n, row_bytes;
    uint ldy; // y row stride in elements
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

// Threadgroups of 4 simdgroups (128 threads); grid (ceil(n / 64), ceil(t / 64)). k must be a multiple of 64.
template <int Type>
kernel void gemm(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                 constant GemmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                 ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup half xs[TM * TK], ws[TN * TK];
    const uint n0 = tg.x * TN, t0 = tg.y * TM;
    // Thread tid stages 32 values of tile row tid / 2 from column (tid % 2) * 32.
    const uint r = tid / 2, c = (tid % 2) * 32;

    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tx(xs, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw(ws, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;

    for (uint k = 0; k < a.k; k += TK) {
        threadgroup half* xo = xs + r * TK + c;
        if (t0 + r < a.t) {
            device const float4* xp = reinterpret_cast<device const float4*>(x + uint64_t(t0 + r) * a.k + k + c);
            for (short i = 0; i < 8; ++i) {
                const float4 v = xp[i];
                xo[4 * i] = half(v.x);
                xo[4 * i + 1] = half(v.y);
                xo[4 * i + 2] = half(v.z);
                xo[4 * i + 3] = half(v.w);
            }
        } else {
            for (short i = 0; i < 32; ++i)
                xo[i] = 0.0h;
        }
        threadgroup half* wo = ws + r * TK + c;
        if (n0 + r < a.n) {
            dequant32<Type>(w + uint64_t(n0 + r) * a.row_bytes, k + c, wo);
        } else {
            for (short i = 0; i < 32; ++i)
                wo[i] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tx, tw, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<device float, dextents<int32_t, 2>, tensor_inline> ty(y, dextents<int32_t, 2>(a.n, a.t),
                                                                 array<int32_t, 2>{1, int32_t(a.ldy)});
    acc.store(ty.slice(n0, t0));
}

template [[host_name("gemm_f32")]] kernel void gemm<F32>(device const uchar*, device const float*, device float*,
                                                         constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0")]] kernel void gemm<Q8_0>(device const uchar*, device const float*, device float*,
                                                           constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k")]] kernel void gemm<Q6_K>(device const uchar*, device const float*, device float*,
                                                           constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xxs")]] kernel void gemm<IQ2_XXS>(device const uchar*, device const float*,
                                                                 device float*, constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s")]] kernel void gemm<IQ2_S>(device const uchar*, device const float*, device float*,
                                                             constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs")]] kernel void gemm<IQ3_XXS>(device const uchar*, device const float*,
                                                                 device float*, constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f16")]] kernel void gemm<F16>(device const uchar*, device const float*, device float*,
                                                         constant GemmArgs&, uint2, ushort);

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
