// Prefill GEMM: y[t][n] = sum_k W[n][k] * x[t][k] for many tokens, x in half (F32, Q8_0, Q6_K, Q5_K, Q4_K, IQ1_S, IQ1_M,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, F16 weights); matmul2d accumulates into a float cooperative tensor (the GPU's matrix units where it has
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

enum WeightType { F32 = 0, Q8_0 = 1, Q6_K = 2, IQ2_XXS = 3, IQ2_S = 4, IQ3_XXS = 5, F16 = 6, IQ3_S = 7, Q4_K = 8, Q5_K = 9, IQ4_XS = 10, IQ2_XS = 11, IQ1_S = 12, IQ1_M = 13 };

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
    } else if (Type == IQ2_XS) {
        // Per 32: four 16-bit entries (9-bit grid index, 7-bit sign index), a 4-bit scale per 16.
        device const block_iq2_xs* blk = reinterpret_cast<device const block_iq2_xs*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32;
        device const ushort* q2 = blk->qs + 4 * ib;
        const float d = float(blk->d);
        for (short l = 0; l < 4; ++l) {
            const float dl = d * (0.5f + float((blk->scales[ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
            grid8(reinterpret_cast<constant uchar*>(iq2xs_grid + (q2[l] & 511)), ksigns_iq2xs[q2[l] >> 9], dl,
                  out + 8 * l);
        }
    } else if (Type == IQ1_S || Type == IQ1_M) {
        // Per 32: four 11-bit indices into a grid of 8 values in 0..2 (nibbles), times an odd 3-bit scale, plus a
        // shift of -1 +- delta. IQ1_S: d per 256, one scale and shift sign per 32 in qh; IQ1_M: d spread over the
        // scales' top nibbles, a scale per 16 and a shift sign per 8.
        const uint ib = (k0 % 256) / 32;
        constexpr float delta = 0.125f;
        if (Type == IQ1_S) {
            device const block_iq1_s* blk = reinterpret_cast<device const block_iq1_s*>(row) + k0 / 256;
            const uint qh = blk->qh[ib];
            const float dl = float(blk->d) * float(2 * ((qh >> 12) & 7) + 1);
            const float ml = dl * ((qh & 0x8000) ? -1 - delta : -1 + delta);
            for (short l = 0; l < 4; ++l) {
                constant uchar* g = reinterpret_cast<constant uchar*>(iq1s_grid_gpu + (blk->qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)));
                for (short i = 0; i < 4; ++i) {
                    out[8 * l + i] = half(dl * float(g[i] & 0xf) + ml);
                    out[8 * l + 4 + i] = half(dl * float(g[i] >> 4) + ml);
                }
            }
        } else {
            device const block_iq1_m* blk = reinterpret_cast<device const block_iq1_m*>(row) + k0 / 256;
            device const ushort* sc = reinterpret_cast<device const ushort*>(blk->scales);
            const ushort u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const float d = float(as_type<half>(u));
            for (short l = 0; l < 4; ++l) {
                const uint half16 = l / 2, qh = blk->qh[2 * ib + half16];
                const float dl = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * half16)) & 7) + 1);
                const uint bits = (l & 1) ? (qh >> 4) : qh; // index bits 0-2, shift sign bit 3
                const float ml = dl * ((bits & 8) ? -1 - delta : -1 + delta);
                constant uchar* g = reinterpret_cast<constant uchar*>(iq1s_grid_gpu + (blk->qs[4 * ib + l] | ((bits & 7) << 8)));
                for (short i = 0; i < 4; ++i) {
                    out[8 * l + i] = half(dl * float(g[i] & 0xf) + ml);
                    out[8 * l + 4 + i] = half(dl * float(g[i] >> 4) + ml);
                }
            }
        }
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
    } else if (Type == IQ3_S) {
        // Per 32: eight index bytes with a ninth bit each in qh, four sign bytes, a 4-bit scale (1 + 2s).
        device const block_iq3_s* blk = reinterpret_cast<device const block_iq3_s*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32, qh = blk->qh[ib];
        device const uchar* qs = blk->qs + 8 * ib;
        device const uchar* sg = blk->signs + 4 * ib;
        const float dl = float(blk->d) * float(1 + 2 * ((blk->scales[ib / 2] >> (4 * (ib % 2))) & 0xf));
        for (short l = 0; l < 4; ++l) {
            constant uchar* g1 = reinterpret_cast<constant uchar*>(iq3s_grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
            constant uchar* g2 = reinterpret_cast<constant uchar*>(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
            for (short i = 0; i < 4; ++i) {
                out[8 * l + i] = half(dl * float(g1[i]) * ((sg[l] >> i) & 1 ? -1.0f : 1.0f));
                out[8 * l + 4 + i] = half(dl * float(g2[i]) * ((sg[l] >> (i + 4)) & 1 ? -1.0f : 1.0f));
            }
        }
    } else if (Type == IQ4_XS) {
        // Per 32: a 6-bit scale (low 4 bits in scales_l, high 2 in scales_h), 16 bytes of nibbles into the
        // non-linear table: low nibbles are weights 0-15, high nibbles 16-31.
        device const block_iq4_xs* blk = reinterpret_cast<device const block_iq4_xs*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32;
        const int ls = int(((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4));
        const float dl = float(blk->d) * float(ls - 32);
        device const uchar* qs = blk->qs + 16 * ib;
        for (short l = 0; l < 16; ++l) {
            out[l] = half(dl * float(kvalues_iq4nl[qs[l] & 0xf]));
            out[l + 16] = half(dl * float(kvalues_iq4nl[qs[l] >> 4]));
        }
    } else if (Type == Q4_K || Type == Q5_K) {
        // Per 256: d, dmin, 6-bit scales and mins of 8 sub-blocks of 32, (Q5_K) 32 bytes of fifth bits, then the
        // nibbles: sub-block j is the low (j even) or high nibbles of bytes 32 * (j / 2)..; its fifth bit is bit j.
        constexpr uint bytes = Type == Q4_K ? 144 : 176;
        device const uchar* blk = row + (k0 / 256) * bytes;
        const uint j = (k0 % 256) / 32;
        device const uchar* sc = blk + 4;
        const uint s = j < 4 ? (sc[j] & 63) : ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        const uint m = j < 4 ? (sc[j + 4] & 63) : ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
        const float d = float(*reinterpret_cast<device const half*>(blk)) * float(s);
        const float dm = float(*reinterpret_cast<device const half*>(blk + 2)) * float(m);
        device const uchar* qh = blk + 16;
        device const uchar* ql = blk + (Type == Q4_K ? 16 : 48) + 32 * (j / 2);
        for (short l = 0; l < 32; ++l) {
            uint q = (j & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
            if (Type == Q5_K)
                q |= ((qh[l] >> j) & 1) << 4;
            out[l] = half(d * float(q) - dm);
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
// memory (fewer, longer matmul2d steps run much faster than 64-wide ones), IQ weights in tiles of up to 32 tokens
// 64 at a time (measured 12-23% faster there; even or slower from 64 tokens). k must be a multiple of 32.
template <int Type, int TM, int TN>
kernel void gemm(device const uchar* w [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]],
                 constant GemmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                 ushort tid [[thread_index_in_threadgroup]])
{
    constexpr bool iq = Type == IQ1_S || Type == IQ1_M || Type == IQ2_XXS || Type == IQ2_XS || Type == IQ2_S ||
                        Type == IQ3_XXS || Type == IQ3_S;
    constexpr int KD = iq && TM <= 32 ? 64 : 256, NSG = 4;
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

// Few-token GEMM for IQ2_XXS, IQ2_S and IQ3_XXS weights: each weight is read and dequantized once per NT tokens, in
// registers, and multiplied with every token (the tiled GEMM above dequantizes into threadgroup memory and runs a
// 32-token tile however few tokens there are). Simdgroups of FR rows, FSG to a threadgroup; lane l takes the 32-weight
// pieces l, l + 32, ... of each row. Grid: ceil(n / (FSG * FR)) threadgroups.
constant short FR = 4, FSG = 4;

// One 32-weight piece's header, then its 8-weight groups g = 0..3 (table copies in threadgroup memory).
template <int Type> struct Piece;

template <> struct Piece<IQ2_XXS> {
    uint idx, aux;
    float dl;
    void load(device const uchar* row, uint k0)
    {
        device const block_iq2_xxs* blk = reinterpret_cast<device const block_iq2_xxs*>(row) + k0 / 256;
        device const ushort* q2 = blk->qs + 4 * ((k0 % 256) / 32);
        idx = uint(q2[0]) | (uint(q2[1]) << 16);
        aux = uint(q2[2]) | (uint(q2[3]) << 16);
        dl = float(blk->d) * (0.5f + float(aux >> 28)) * 0.25f;
    }
    void group(short g, threadgroup const uchar* grid, threadgroup const uchar* signs, thread float* out) const
    {
        threadgroup const uchar* e = grid + 8 * ((idx >> (8 * g)) & 255);
        const uint sb = signs[(aux >> (7 * g)) & 127];
        for (short i = 0; i < 8; ++i)
            out[i] = dl * float(e[i]) * ((sb >> i) & 1 ? -1.0f : 1.0f);
    }
};

template <> struct Piece<IQ2_S> {
    uint qs, sg, qh;
    float d0, d1; // scales of the first and second 16 weights
    void load(device const uchar* row, uint k0)
    {
        device const block_iq2_s* blk = reinterpret_cast<device const block_iq2_s*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32;
        device const uchar* q = blk->qs + 4 * ib;
        device const uchar* s = blk->qs + 32 + 4 * ib;
        qs = uint(q[0]) | (uint(q[1]) << 8) | (uint(q[2]) << 16) | (uint(q[3]) << 24);
        sg = uint(s[0]) | (uint(s[1]) << 8) | (uint(s[2]) << 16) | (uint(s[3]) << 24);
        qh = blk->qh[ib];
        const float d = float(blk->d);
        d0 = d * (0.5f + float(blk->scales[ib] & 0xf)) * 0.25f;
        d1 = d * (0.5f + float(blk->scales[ib] >> 4)) * 0.25f;
    }
    void group(short g, threadgroup const uchar* grid, threadgroup const uchar*, thread float* out) const
    {
        threadgroup const uchar* e = grid + 8 * (((qs >> (8 * g)) & 255) | (((qh >> (2 * g)) & 3) << 8));
        const uint sb = (sg >> (8 * g)) & 255;
        const float dl = g < 2 ? d0 : d1;
        for (short i = 0; i < 8; ++i)
            out[i] = dl * float(e[i]) * ((sb >> i) & 1 ? -1.0f : 1.0f);
    }
};

template <> struct Piece<IQ3_XXS> {
    uint q_lo, q_hi, aux; // eight grid indices (bytes), signs and scale
    float dl;
    void load(device const uchar* row, uint k0)
    {
        device const block_iq3_xxs* blk = reinterpret_cast<device const block_iq3_xxs*>(row) + k0 / 256;
        const uint ib = (k0 % 256) / 32;
        device const uchar* q3 = blk->qs + 8 * ib;
        q_lo = uint(q3[0]) | (uint(q3[1]) << 8) | (uint(q3[2]) << 16) | (uint(q3[3]) << 24);
        q_hi = uint(q3[4]) | (uint(q3[5]) << 8) | (uint(q3[6]) << 16) | (uint(q3[7]) << 24);
        device const ushort* gas = reinterpret_cast<device const ushort*>(blk->qs + 64) + 2 * ib;
        aux = uint(gas[0]) | (uint(gas[1]) << 16);
        dl = float(blk->d) * (0.5f + float(aux >> 28)) * 0.5f;
    }
    void group(short g, threadgroup const uchar* grid, threadgroup const uchar* signs, thread float* out) const
    {
        const uint q = g < 2 ? q_lo : q_hi, sh = 16 * (g & 1);
        threadgroup const uchar* g1 = grid + 4 * ((q >> sh) & 255);
        threadgroup const uchar* g2 = grid + 4 * ((q >> (sh + 8)) & 255);
        const uint sb = signs[(aux >> (7 * g)) & 127];
        for (short i = 0; i < 4; ++i) {
            out[i] = dl * float(g1[i]) * ((sb >> i) & 1 ? -1.0f : 1.0f);
            out[4 + i] = dl * float(g2[i]) * ((sb >> (i + 4)) & 1 ? -1.0f : 1.0f);
        }
    }
};

template <int Type, int NT>
kernel void gemm_few(device const uchar* w [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int GRID_BYTES = Type == IQ2_S ? 1024 * 8 : Type == IQ2_XXS ? 256 * 8 : 256 * 4;
    threadgroup uchar grid[GRID_BYTES];
    threadgroup uchar signs[128];
    {
        constant uchar* src = Type == IQ2_S     ? reinterpret_cast<constant uchar*>(iq2s_grid)
                              : Type == IQ2_XXS ? reinterpret_cast<constant uchar*>(iq2xxs_grid)
                                                : reinterpret_cast<constant uchar*>(iq3xxs_grid);
        for (uint i = tid; i < GRID_BYTES / 4; i += FSG * 32)
            reinterpret_cast<threadgroup uint*>(grid)[i] = reinterpret_cast<constant uint*>(src)[i];
        if (tid < 128)
            signs[tid] = ksigns_iq2xs[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r0 = (tg * FSG + sg) * FR;
    if (r0 >= a.n)
        return;
    const uint pieces = a.k / 32;
    for (uint t0 = 0; t0 < a.t; t0 += NT) {
        const uint nt = min(uint(NT), a.t - t0);
        float acc[FR][NT];
        for (short r = 0; r < FR; ++r)
            for (short t = 0; t < NT; ++t)
                acc[r][t] = 0.0f;
        for (uint p = lane; p < pieces; p += 32) {
            const uint k0 = p * 32;
            Piece<Type> pc[FR];
            for (short r = 0; r < FR; ++r)
                pc[r].load(w + uint64_t(min(r0 + r, a.n - 1)) * a.row_bytes, k0);
            for (short g = 0; g < 4; ++g) {
                float wv[FR][8];
                for (short r = 0; r < FR; ++r)
                    pc[r].group(g, grid, signs, wv[r]);
                for (short t = 0; t < NT; ++t) {
                    if (uint(t) >= nt)
                        break;
                    device const half4* xp =
                        reinterpret_cast<device const half4*>(x + uint64_t(t0 + t) * a.ldx + k0 + 8 * g);
                    const float4 x0 = float4(xp[0]), x1 = float4(xp[1]);
                    for (short r = 0; r < FR; ++r)
                        acc[r][t] += dot(float4(wv[r][0], wv[r][1], wv[r][2], wv[r][3]), x0) +
                                     dot(float4(wv[r][4], wv[r][5], wv[r][6], wv[r][7]), x1);
                }
            }
        }
        for (short r = 0; r < FR; ++r)
            for (short t = 0; t < NT; ++t) {
                const float s = simd_sum(acc[r][t]);
                if (lane == 0 && uint(t) < nt && r0 + r < a.n)
                    y[uint64_t(t0 + t) * a.ldy + r0 + r] = s;
            }
    }
}

#define GEMM_FEW(name, type, nt)                                                                                      \
    template [[host_name("gemm_few_" name "_" #nt)]] kernel void gemm_few<type, nt>(                                 \
        device const uchar*, device const half*, device float*, constant GemmArgs&, uint, ushort, ushort, ushort);
#define GEMM_FEW_ALL(name, type) GEMM_FEW(name, type, 1) GEMM_FEW(name, type, 2) GEMM_FEW(name, type, 4)
GEMM_FEW_ALL("iq2_xxs", IQ2_XXS)
GEMM_FEW_ALL("iq2_s", IQ2_S)
GEMM_FEW_ALL("iq3_xxs", IQ3_XXS)

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
template [[host_name("gemm_iq2_xxs_16x32")]] kernel void gemm<IQ2_XXS, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s_16x32")]] kernel void gemm<IQ2_S, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs_16x32")]] kernel void gemm<IQ3_XXS, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_s_64x32")]] kernel void gemm<IQ3_S, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_s_32x32")]] kernel void gemm<IQ3_S, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_s_16x32")]] kernel void gemm<IQ3_S, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xxs_128x32")]] kernel void gemm<IQ2_XXS, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s_128x32")]] kernel void gemm<IQ2_S, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs_128x32")]] kernel void gemm<IQ3_XXS, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_s_128x32")]] kernel void gemm<IQ3_S, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xxs_256x32")]] kernel void gemm<IQ2_XXS, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_s_256x32")]] kernel void gemm<IQ2_S, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_xxs_256x32")]] kernel void gemm<IQ3_XXS, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq3_s_256x32")]] kernel void gemm<IQ3_S, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f32_128x32")]] kernel void gemm<F32, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f32_256x32")]] kernel void gemm<F32, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0_128x32")]] kernel void gemm<Q8_0, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0_256x32")]] kernel void gemm<Q8_0, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k_128x32")]] kernel void gemm<Q6_K, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k_256x32")]] kernel void gemm<Q6_K, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f16_128x32")]] kernel void gemm<F16, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_f16_256x32")]] kernel void gemm<F16, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q4_k_32x32")]] kernel void gemm<Q4_K, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q4_k_64x32")]] kernel void gemm<Q4_K, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q4_k_128x32")]] kernel void gemm<Q4_K, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q4_k_256x32")]] kernel void gemm<Q4_K, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xs_16x32")]] kernel void gemm<IQ2_XS, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xs_32x32")]] kernel void gemm<IQ2_XS, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xs_64x32")]] kernel void gemm<IQ2_XS, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xs_128x32")]] kernel void gemm<IQ2_XS, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq2_xs_256x32")]] kernel void gemm<IQ2_XS, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_s_16x32")]] kernel void gemm<IQ1_S, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_s_32x32")]] kernel void gemm<IQ1_S, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_s_64x32")]] kernel void gemm<IQ1_S, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_s_128x32")]] kernel void gemm<IQ1_S, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_s_256x32")]] kernel void gemm<IQ1_S, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_m_16x32")]] kernel void gemm<IQ1_M, 16, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_m_32x32")]] kernel void gemm<IQ1_M, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_m_64x32")]] kernel void gemm<IQ1_M, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_m_128x32")]] kernel void gemm<IQ1_M, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq1_m_256x32")]] kernel void gemm<IQ1_M, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq4_xs_32x32")]] kernel void gemm<IQ4_XS, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq4_xs_64x32")]] kernel void gemm<IQ4_XS, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq4_xs_128x32")]] kernel void gemm<IQ4_XS, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_iq4_xs_256x32")]] kernel void gemm<IQ4_XS, 256, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q5_k_32x32")]] kernel void gemm<Q5_K, 32, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q5_k_64x32")]] kernel void gemm<Q5_K, 64, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q5_k_128x32")]] kernel void gemm<Q5_K, 128, 32>(device const uchar*, device const half*, device float*,
    constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q5_k_256x32")]] kernel void gemm<Q5_K, 256, 32>(device const uchar*, device const half*, device float*,
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
