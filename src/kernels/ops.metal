// Generic transformer pieces any model can use: attention heads in half tiles (with rotary positions), masked flash
// attention with grouped key/value heads, activations. Activations are fp32 rows [token][width] unless noted.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#include "rows.h"

struct HeadsArgs {
    uint rows;     // tokens
    uint ld;       // x row stride in elements
    uint col0;     // the first head's column
    uint dim;      // per head
    uint n_pad;    // rows of a head in y
    uint row0;     // y row of token 0
    uint rope;     // 0 none; 1 halves (i with i + dim/2); 2 adjacent pairs (2i with 2i + 1)
};

// y[h][row0 + t][d] = half(x[t][col0 + h * dim + d]), rotated by the angle table cs[t][i] = (cos, sin) of pair i when
// rope is set. Grid (dim / 2, rows, heads).
kernel void heads_half(device const float* x [[buffer(0)]], device const float2* cs [[buffer(1)]],
                       device half* y [[buffer(2)]], constant HeadsArgs& a [[buffer(3)]],
                       uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y, h = gid.z;
    if (i >= a.dim / 2 || t >= a.rows)
        return;
    device const float* xr = x + uint64_t(t) * a.ld + a.col0 + h * a.dim;
    device half* yr = y + (uint64_t(h) * a.n_pad + a.row0 + t) * a.dim;
    const uint i0 = a.rope == 2 ? 2 * i : i, i1 = a.rope == 2 ? 2 * i + 1 : i + a.dim / 2;
    float u = xr[i0], v = xr[i1];
    if (a.rope) {
        const float2 c = cs[uint64_t(t) * (a.dim / 2) + i];
        const float ru = u * c.x - v * c.y, rv = v * c.x + u * c.y;
        u = ru;
        v = rv;
    }
    yr[i0] = half(u);
    yr[i1] = half(v);
}

// Zero rows [from, n_pad) of every head of a half tile, so padded keys and values read as zeros.
kernel void heads_clear(device half* y [[buffer(0)]], constant uint4& a [[buffer(1)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const uint dim = a.x, n_pad = a.y, from = a.z;
    const uint d = gid.x, t = from + gid.y, h = gid.z;
    if (d < dim && t < n_pad)
        y[(uint64_t(h) * n_pad + t) * dim + d] = 0.0h;
}

struct AttnArgs {
    uint n_q;      // queries
    uint q_pad;    // rows of a query head (a multiple of 32)
    uint n_k;      // keys
    uint k_pad;    // rows of a key / value head (a multiple of 128)
    uint heads, kv_heads;
    uint q_pos0;   // the position of query 0 among the keys
    uint block;    // positions from here on see each other both ways (the image); below, causal
    uint width;    // o row length
    float scale;
};

// x = gelu(x), the tanh approximation.
kernel void gelu_tanh(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    if (i >= n)
        return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// x = x * sigmoid(x).
kernel void silu(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        x[i] = x[i] / (1.0f + exp(-x[i]));
}

// Rows [0, rows) of every head between half tiles of different heights: dst[h][r][d] = src[h][r][d].
// Grid (dim, rows, heads).
struct HeadRowsArgs {
    uint dim, src_pad, dst_pad, rows;
};
kernel void heads_rows(device const half* src [[buffer(0)]], device half* dst [[buffer(1)]],
                       constant HeadRowsArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, r = gid.y, h = gid.z;
    if (d < a.dim && r < a.rows)
        dst[(uint64_t(h) * a.dst_pad + r) * a.dim + d] = src[(uint64_t(h) * a.src_pad + r) * a.dim + d];
}

// o[q][h * D + d] = sum_j softmax_j(scale * q . k_j) v_j[d] over the keys j query q may see: j <= its position, or
// both at or past `block`. q half [heads][q_pad][D], k and v half [kv_heads][k_pad][D] (head h reads kv head
// h / (heads / kv_heads)). Each simdgroup works alone over its R query rows, with no threadgroup barriers, so the
// simdgroups' products and softmaxes overlap: the scores of K keys at a time go to its own threadgroup memory for the
// running softmax (32 / R lanes a row, in base 2), the output stays in registers and both products run through
// matmul2d with float accumulation. A row's reference maximum moves only when a score passes it by 8 (the
// exponentials stay within 256, half's range), so most passes leave the output unscaled. q_pad a multiple of
// R * NSG, k_pad of K. Grid (q_pad / (R * NSG), heads), NSG simdgroups.
// Measured at 32 heads, 4096 queries x 4122 keys, D 128: R 16, K 32, 4 simdgroups 7.1 ms; 16 x 64 x 4 7.2; with
// 8 or 32 rows a simdgroup 30-36 ms; the earlier kernel, 32 queries a threadgroup in lockstep (two threadgroup
// barriers a pass), 12.0.
template <int D, int R, int K, int NSG>
kernel void flash_attn_sg(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                          device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                          constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    constexpr int TPR = 32 / R, CPT = K / TPR;
    threadgroup float s_all[NSG][R * K];
    threadgroup half p_all[NSG][R * K];
    threadgroup float scale_all[NSG][R];
    threadgroup float* s = s_all[sg];
    threadgroup half* p = p_all[sg];
    threadgroup float* row_scale = scale_all[sg];
    const uint q0 = (tg.x * NSG + sg) * R, h = tg.y, kvh = h / (a.heads / a.kv_heads);
    if (q0 >= a.n_q)
        return;
    const uint64_t qhead = uint64_t(h) * a.q_pad * D, khead = uint64_t(kvh) * a.k_pad * D;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qhead + uint64_t(q0) * D,
                                                                dextents<int32_t, 2>(D, R));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(const_cast<device half*>(v) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> ts(s, dextents<int32_t, 2>(K, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(p, dextents<int32_t, 2>(K, R));
    constexpr auto qk_desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(R, D, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<1>> qk;
    matmul2d<pv_desc, execution_simdgroups<1>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    const uint r = lane / TPR, c0 = (lane % TPR) * CPT;
    const uint pos = a.q_pos0 + q0 + r;
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m = -INFINITY, l = 0.0f;
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < sc.get_capacity(); ++i)
            if (sc.is_valid_element(i))
                sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        sc.store(ts);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float x[CPT], mt = -INFINITY;
        for (uint c = 0; c < CPT; ++c) {
            const uint j = j0 + c0 + c;
            const bool seen = j < a.n_k && (j <= pos || (pos >= a.block && j >= a.block));
            x[c] = seen ? s[r * K + c0 + c] * scale : -INFINITY;
            mt = max(mt, x[c]);
        }
        for (ushort mm = 1; mm < TPR; mm *= 2)
            mt = max(mt, simd_shuffle_xor(mt, mm));
        float f = 1.0f;
        if (mt > m + 8.0f) {
            f = m == -INFINITY ? 0.0f : exp2(m - mt);
            m = mt;
        }
        const bool rescale = simd_any(f != 1.0f);
        float sum = 0.0f;
        for (uint c = 0; c < CPT; ++c) {
            const float e = m == -INFINITY ? 0.0f : exp2(x[c] - m);
            p[r * K + c0 + c] = half(e);
            sum += e;
        }
        for (ushort mm = 1; mm < TPR; mm *= 2)
            sum += simd_shuffle_xor(sum, mm);
        l = l * f + sum;
        if (lane % TPR == 0)
            row_scale[r] = f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale)
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                if (acc.is_valid_element(i))
                    acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
        auto vt = tv.slice(0, int(j0));
        pv.run(tp, vt, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane % TPR == 0)
        row_scale[r] = l > 0.0f ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}

#define FLASH_ATTN_SG(d, r, k, nsg)                                                                                   \
    template [[host_name("flash_attn_sg_" #d "_" #r "x" #k "_" #nsg)]] kernel void flash_attn_sg<d, r, k, nsg>(      \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, \
        ushort);
FLASH_ATTN_SG(128, 16, 32, 4)
// 64 keys a pass: the same at 4k keys, 12% faster at 19k (384 -> 336 ms for 56 heads); the runtime's 64 and 128-wide
// heads now run flash_attn_coop below (this one is the bench's reference), 256-wide ones flash_attn_sg_256_16x32_4.
FLASH_ATTN_SG(128, 16, 64, 4)
FLASH_ATTN_SG(256, 16, 32, 4)
FLASH_ATTN_SG(64, 16, 32, 4)

// Flash attention with the four simdgroups of a threadgroup cooperating on 64 query rows: Q K^T and P V are one
// cooperative matmul2d each over 64-row tiles, so every key and value row is read by two simdgroups instead of four
// (matmul2d itself runs 16 x 64 and 64 x 64 tiles at the same 60-65 TFLOP/s; 16 x 128 at 39). A cooperative 64 x N
// result is a 2 x 2 grid
// of 32 x N/2 blocks over the simdgroups (sg >> 1 picks the 16-row sub-block of each 32-row half, sg & 1 the 16-column
// sub-block of each 32-column quarter); within a block lane l holds rows r0 = ((l >> 1) & 3) | ((l >> 4) << 2),
// r0 + 8, r0 + 32, r0 + 40 and, in each 32 columns, the 4 from 4 ((l & 1) + 2 ((l >> 3) & 1)): element i is column
// quarter (i >> 3) % (K / 32), row (i >> 2) & 1 and i / (K / 4) over those, column i & 3. The four lanes of a row in a
// simdgroup differ in bits 0 and 3, so a row's maximum takes two shuffles; its other half sits in the partner
// simdgroup (sg ^ 1), exchanged through threadgroup memory once a pass. The scores stay in registers (no score
// buffer); only P (half) goes through threadgroup memory, double-buffered: two threadgroup barriers a pass. Sums are
// kept per simdgroup and added at the end. 32 keys a pass (64 spilled). 56 heads of 128 at 19308 keys: 384 ms with
// flash_attn_sg_128_16x32_4, 336 with 64 keys a pass, 302 here; 4096 x 4122 with 32 heads 7.3 -> 6.6 ms; 32 heads
// of 64 at 1797 keys 3.0 -> 2.2 ms. The same arguments and grid as flash_attn_sg with 64 queries a threadgroup.
template <int D, int K>
kernel void flash_attn_coop(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                            device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                            constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here
    constexpr int R = 64, NSG = 4, CAP = R * K / 128, ACAP = R * D / 128;
    constexpr int CQ = K / 32, AQ = D / 32; // column quarters of the score and output tiles
    threadgroup half p_all[2][R * K];
    threadgroup float pmax[2][2][R];
    threadgroup float lsum[2][R];
    const uint q0 = tg.x * R, h = tg.y, kvh = h / (a.heads / a.kv_heads);
    if (q0 >= a.n_q)
        return;
    const uint64_t qhead = uint64_t(h) * a.q_pad * D, khead = uint64_t(kvh) * a.k_pad * D;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qhead + uint64_t(q0) * D,
                                                                dextents<int32_t, 2>(D, R));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(const_cast<device half*>(v) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp0(p_all[0], dextents<int32_t, 2>(K, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp1(p_all[1], dextents<int32_t, 2>(K, R));
    constexpr auto qk_desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(R, D, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp0), decltype(vt0), float>();
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] = 0.0f;
    const uint sgx = sg & 1, sgy = sg >> 1;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2) | (sgy << 4); // this lane's rows: r0 + {0, 8, 32, 40}
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1)) + 16 * sgx; // within each 32 columns
    const bool writer = (lane & 9) == 0;                                      // one lane a row for the exchanges
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint buf = 0;
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < CAP; ++i)
            sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        // Scaled scores, masked where the pass reaches keys a row may not see; this lane's maxima per row.
        const bool plain = j0 + K <= a.n_k && ((a.q_pos0 + q0 >= a.block && j0 >= a.block) ||
                                               j0 + K - 1 <= a.q_pos0 + q0);
        float mt[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * CQ)) << 1);
            float x = sc[i] * scale;
            if (!plain) {
                const uint j = j0 + ((i >> 3) % CQ) * 32 + c_base + (i & 3);
                const uint pos = a.q_pos0 + q0 + r0 + (slot & 1) * 8 + (slot >> 1) * 32;
                if (!(j < a.n_k && (j <= pos || (pos >= a.block && j >= a.block))))
                    x = -INFINITY;
            }
            sc[i] = x;
            mt[slot] = max(mt[slot], x);
        }
        for (int s = 0; s < 4; ++s) {
            mt[s] = max(mt[s], simd_shuffle_xor(mt[s], 1));
            mt[s] = max(mt[s], simd_shuffle_xor(mt[s], 8));
        }
        if (writer)
            for (int s = 0; s < 4; ++s)
                pmax[buf][sgx][r0 + (s & 1) * 8 + (s >> 1) * 32] = mt[s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The reference maximum moves only when a score passes it by 8 (the exponentials stay within 256, half's
        // range), so most passes leave the output unscaled.
        bool rescale = false;
        float f[4];
        for (int s = 0; s < 4; ++s) {
            mt[s] = max(mt[s], pmax[buf][sgx ^ 1][r0 + (s & 1) * 8 + (s >> 1) * 32]);
            f[s] = 1.0f;
            if (mt[s] > m[s] + 8.0f) {
                f[s] = m[s] == -INFINITY ? 0.0f : exp2(m[s] - mt[s]);
                m[s] = mt[s];
                l[s] *= f[s];
                rescale = true;
            }
        }
        if (simd_any(rescale))
            for (uint16_t i = 0; i < ACAP; ++i)
                acc[i] *= f[((i >> 2) & 1) | ((i / (8 * AQ)) << 1)];
        threadgroup half* p = p_all[buf];
        for (uint16_t i = 0; i < CAP; i += 4) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * CQ)) << 1);
            const uint row = r0 + (slot & 1) * 8 + (slot >> 1) * 32, col = ((i >> 3) % CQ) * 32 + c_base;
            const float mm = m[slot];
            half4 e;
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                const float ex = mm == -INFINITY ? 0.0f : exp2(sc[i + c] - mm);
                e[c] = half(ex);
                sum += ex;
            }
            *reinterpret_cast<threadgroup half4*>(p + row * K + col) = e;
            l[slot] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto vt = tv.slice(0, int(j0));
        if (buf == 0)
            pv.run(tp0, vt, acc);
        else
            pv.run(tp1, vt, acc);
        buf ^= 1;
    }
    // The rows' sums: this simdgroup's plus the partner's (the four lanes of a row hold the same partial).
    for (int s = 0; s < 4; ++s) {
        l[s] += simd_shuffle_xor(l[s], 1);
        l[s] += simd_shuffle_xor(l[s], 8);
    }
    if (writer)
        for (int s = 0; s < 4; ++s)
            lsum[sgx][r0 + (s & 1) * 8 + (s >> 1) * 32] = l[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[4];
    for (int s = 0; s < 4; ++s) {
        const float total = l[s] + lsum[sgx ^ 1][r0 + (s & 1) * 8 + (s >> 1) * 32];
        inv[s] = total > 0.0f ? 1.0f / total : 0.0f;
    }
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] *= inv[((i >> 2) & 1) | ((i / (8 * AQ)) << 1)];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}
#define FLASH_ATTN_COOP(d, k)                                                                                        \
    template [[host_name("flash_attn_coop_" #d "_" #k)]] kernel void flash_attn_coop<d, k>(                           \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, \
        ushort);
FLASH_ATTN_COOP(128, 32)
FLASH_ATTN_COOP(64, 32)

struct HeadsNormArgs {
    uint ld, col0, dim, n_pad, row0, rope; // as HeadsArgs
    float eps;
};

// heads_half with each head's values RMS-normed first (times w[dim]): one threadgroup of dim / 2 threads per
// (token, head). Grid (rows, heads) threadgroups.
kernel void heads_norm_half(device const float* x [[buffer(0)]], device const float2* cs [[buffer(1)]],
                            device half* y [[buffer(2)]], device const float* w [[buffer(3)]],
                            constant HeadsNormArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            ushort i [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[8];
    const uint t = tg.x, h = tg.y;
    device const float* xr = x + uint64_t(t) * a.ld + a.col0 + h * a.dim;
    const uint i0 = a.rope == 2 ? 2 * i : i, i1 = a.rope == 2 ? 2 * i + 1 : i + a.dim / 2;
    float u = xr[i0], v = xr[i1];
    const float sq = simd_sum(u * u + v * v);
    if (lane == 0)
        part[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < a.dim / 64; ++s)
        total += part[s];
    const float inv = rsqrt(total / float(a.dim) + a.eps);
    u *= inv * w[i0];
    v *= inv * w[i1];
    if (a.rope) {
        const float2 c = cs[uint64_t(t) * (a.dim / 2) + i];
        const float ru = u * c.x - v * c.y, rv = v * c.x + u * c.y;
        u = ru;
        v = rv;
    }
    device half* yr = y + (uint64_t(h) * a.n_pad + a.row0 + t) * a.dim;
    yr[i0] = half(u);
    yr[i1] = half(v);
}

// m[t][i] = silu(gu[t][i]) * gu[t][n + i]: gate and up side by side in a row. Grid (n, rows).
kernel void swiglu_rows(device const float* gu [[buffer(0)]], device float* m [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= n)
        return;
    const float g = gu[uint64_t(gid.y) * 2 * n + gid.x], u = gu[uint64_t(gid.y) * 2 * n + n + gid.x];
    m[uint64_t(gid.y) * n + gid.x] = g / (1.0f + exp(-g)) * u;
}

// lo = bfloat(x - hi): what rounding x to bfloat (hi) left out, rows ldx floats apart. Grid (cols, rows).
kernel void bf16_lo(device const float* x [[buffer(0)]], device const bfloat* hi [[buffer(1)]],
                    device bfloat* lo [[buffer(2)]], constant uint2& a [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.x)
        return;
    const uint64_t o = uint64_t(gid.y) * a.x + gid.x;
    lo[o] = bfloat(x[uint64_t(gid.y) * a.y + gid.x] - float(hi[o]));
}

// y[r][i] += z[r][i] with y's rows ldy apart and z's dense. Grid (n, rows).
kernel void add_rows(device float* y [[buffer(0)]], device const float* z [[buffer(1)]], constant uint2& a [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.x)
        y[uint64_t(gid.y) * a.y + gid.x] += z[uint64_t(gid.y) * a.x + gid.x];
}

// y = x / sqrt(mean(x^2) + eps) * w (w of n, or none when wn is 0), one threadgroup per row; n a multiple of 4.
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float4* xr = reinterpret_cast<device const float4*>(x + uint64_t(row) * a.n);
    device float4* yr = reinterpret_cast<device float4*>(y + uint64_t(row) * (a.ystride ? a.ystride : a.n));
    device const float4* w4 = reinterpret_cast<device const float4*>(w);
    const uint n4 = a.n / 4;
    float s = 0.0f;
    for (uint i = tid; i < n4; i += TG) {
        const float4 v = xr[i];
        s += dot(v, v);
    }
    const float inv = rsqrt(tg_sum(s, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < n4; i += TG)
        yr[i] = xr[i] * inv * (a.wn ? w4[i] : float4(1.0f));
}

struct ScanArgs {
    uint head_dim;   // 128
    uint tokens;
    uint qkv_stride; // q, k, v elements between tokens
    uint inner;      // g and o elements between tokens
    uint heads;      // beta elements between tokens
    uint snap;       // nonzero: also write the state after this many tokens to snap_state (for a rollback)
};

// One delta-rule step with per-channel decay. M is the state transposed (M[j][i] = S[i][j]):
//   S[i][:] *= exp(g[i]); delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta
//   S[i][j] += k[i] * delta[j];   o[j] = sum_i S[i][j] q[i] / sqrt(head_dim)
// Threadgroups (heads, head_dim / 32) of 128 threads for rows j0..j0+31 of a head's M (rows are independent: row j
// needs only the shared q, k, g and its own v[j]); four threads per row, each holding columns lane, lane + 4, ...
// of it in registers for all the tokens. head_dim must be 128.
kernel void kda_scan(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                     device const float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                     device float* o [[buffer(6)]], constant ScanArgs& a [[buffer(7)]],
                     device float* snap_state [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint2 tid2 [[thread_position_in_threadgroup]])
{
    constexpr uint D = 128, ROWS = 32, LANES = 4, N = D / LANES;
    threadgroup float qs[D], ks[D], eg[D];
    const uint tid = tid2.x, h = tg.x, j0 = tg.y * ROWS, row = tid / LANES, lane = tid % LANES;
    const uint base = h * D;
    device float* Mg = state + uint64_t(h) * D * D + uint64_t(j0 + row) * D;
    float M[N];
    for (uint n = 0; n < N; ++n)
        M[n] = Mg[lane + n * LANES];
    for (uint t = 0; t < a.tokens; ++t) {
        const uint64_t qo = uint64_t(t) * a.qkv_stride + base, go = uint64_t(t) * a.inner + base;
        qs[tid] = q[qo + tid];
        ks[tid] = k[qo + tid];
        eg[tid] = exp(g[go + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint n = 0; n < N; ++n) {
            M[n] *= eg[lane + n * LANES];
            sum += M[n] * ks[lane + n * LANES];
        }
        sum = quad_sum(sum);
        const float delta = (v[qo + j0 + row] - sum) * beta[t * a.heads + h];
        float out = 0.0f;
        for (uint n = 0; n < N; ++n) {
            M[n] += ks[lane + n * LANES] * delta;
            out += M[n] * qs[lane + n * LANES];
        }
        out = quad_sum(out);
        if (lane == 0)
            o[go + j0 + row] = out * rsqrt(float(D));
        if (t + 1 == a.snap) {
            device float* Ss = snap_state + uint64_t(h) * D * D + uint64_t(j0 + row) * D;
            for (uint n = 0; n < N; ++n)
                Ss[lane + n * LANES] = M[n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint n = 0; n < N; ++n)
        Mg[lane + n * LANES] = M[n];
}

struct SwigluArgs {
    uint n;
    float limit;   // <= 0: no clamp
};

// y = silu(min(g, limit)) * clamp(u, -limit, limit); g and u are [rows][n].
template <typename T>
kernel void swiglu(device const float* gt [[buffer(0)]], device const float* up [[buffer(1)]],
                   device T* y [[buffer(2)]], constant SwigluArgs& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    float g = gt[i], u = up[i];
    if (a.limit > 0.0f) {
        g = min(g, a.limit);
        u = clamp(u, -a.limit, a.limit);
    }
    y[i] = T(g * sigmoidf(g) * u);
}
template [[host_name("swiglu")]] kernel void swiglu<float>(device const float*, device const float*, device float*,
                                                          constant SwigluArgs&, uint);
template [[host_name("swiglu_half")]] kernel void swiglu<half>(device const float*, device const float*, device half*,
                                                              constant SwigluArgs&, uint);

kernel void add(device const float* x [[buffer(0)]], device const float* z [[buffer(1)]],
                device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i] + z[i];
}

// y[t * ldy + i] = x[t * n + i]; grid (n, rows).
kernel void copy_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint2& a [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < a.x)
        y[uint64_t(gid.y) * a.y + gid.x] = x[uint64_t(gid.y) * a.x + gid.x];
}

kernel void copy_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        y[i] = x[i];
}

// y[r][i] += b[i]. Grid (n, rows).
kernel void bias_rows(device float* y [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n)
        y[uint64_t(gid.y) * n + gid.x] += b[gid.x];
}

struct HeadsPartialArgs {
    uint ld, col0; // x rows ld apart; head h at col0 + h * stride
    uint stride;
    uint dim;      // per head (at most 256)
    uint rot;      // the first rot values rotate, i with i + rot / 2 (0: none)
    uint n_pad;    // rows of a head in y
    float eps;     // 0: no norm
    uint mapped;   // row t of y comes from x row map[t] (and its angles), a zero row where map[t] < 0
};

// y[h][t][d] = half of head h of row t, RMS-normed (times w) when eps is set, its first rot values rotated by
// cs[t][i] = (cos, sin). Threadgroups (rows, heads) of dim threads.
kernel void heads_partial(device const float* x [[buffer(0)]], device const float2* cs [[buffer(1)]],
                          device half* y [[buffer(2)]], device const float* w [[buffer(3)]],
                          constant HeadsPartialArgs& a [[buffer(4)]], device const int* map [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float vals[256];
    threadgroup float part[8];
    const uint t = a.mapped ? uint(map[tg.x]) : tg.x, h = tg.y;
    if (a.mapped && map[tg.x] < 0) {
        y[(uint64_t(h) * a.n_pad + tg.x) * a.dim + i] = 0.0h;
        return;
    }
    float v = x[uint64_t(t) * a.ld + a.col0 + h * a.stride + i];
    if (a.eps > 0.0f) {
        const float s = simd_sum(v * v);
        if (lane == 0)
            part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < a.dim / 32; ++j)
            total += part[j];
        v *= rsqrt(total / float(a.dim) + a.eps) * w[i];
    }
    vals[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < a.rot) {
        const uint half_rot = a.rot / 2, p = i % half_rot;
        const float2 c = cs[uint64_t(t) * half_rot + p];
        v = i < half_rot ? vals[i] * c.x - vals[i + half_rot] * c.y : vals[i] * c.x + vals[i - half_rot] * c.y;
    }
    y[(uint64_t(h) * a.n_pad + tg.x) * a.dim + i] = half(v);
}

struct LoraArgs {
    uint k, rows, rank; // the matrix's row length, the rows adapted, the adapter's rank
    uint b_rot;         // row r of the matrix takes B's row (r + b_rot) % rows
    float scale;
};

// W[r][c] += scale * sum_j B[(r + b_rot) % rows][j] * A[j][c] over BF16 (or F16) rows of W (from the adapted ones
// on), in float, rounded back. Grid (k, rows).
template <typename T>
kernel void lora_merge(device T* w [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]],
                       constant LoraArgs& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, r = gid.y;
    if (c >= p.k || r >= p.rows)
        return;
    const uint br = (r + p.b_rot) % p.rows;
    float d = 0.0f;
    for (uint j = 0; j < p.rank; ++j)
        d += b[uint64_t(br) * p.rank + j] * a[uint64_t(j) * p.k + c];
    device T* x = w + uint64_t(r) * p.k + c;
    *x = T(float(*x) + p.scale * d);
}
template [[host_name("lora_merge")]] kernel void lora_merge<bfloat>(device bfloat*, device const float*,
                                                                    device const float*, constant LoraArgs&, uint2);
template [[host_name("lora_merge_f16")]] kernel void lora_merge<half>(device half*, device const float*,
                                                                      device const float*, constant LoraArgs&, uint2);

// The same over int8 rows with a float scale each (gemm_i8's weights, sc[r] the scale of row r): the row's values
// s_r q + scale B A are requantized to the new row maximum. One threadgroup of 256 per adapted row, each thread's
// values in registers: k at most 16384.
kernel void lora_merge_i8(device int8_t* w [[buffer(0)]], device const float* a [[buffer(1)]],
                          device const float* b [[buffer(2)]], constant LoraArgs& p [[buffer(3)]],
                          device float* sc [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float part[8];
    const uint br = (r + p.b_rot) % p.rows;
    device int8_t* wr = w + uint64_t(r) * p.k;
    const float s = sc[r];
    float v[64];
    float m = 0.0f;
    for (uint j = 0, c = tid; c < p.k; ++j, c += 256) {
        float d = 0.0f;
        for (uint q = 0; q < p.rank; ++q)
            d += b[uint64_t(br) * p.rank + q] * a[uint64_t(q) * p.k + c];
        v[j] = s * float(wr[c]) + p.scale * d;
        m = max(m, abs(v[j]));
    }
    m = simd_max(m);
    if ((tid & 31) == 0)
        part[tid >> 5] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (int i = 1; i < 8; ++i)
        m = max(m, part[i]);
    const float ns = m > 0.0f ? m / 127.0f : 1.0f / 127.0f, inv = 1.0f / ns;
    for (uint j = 0, c = tid; c < p.k; ++j, c += 256)
        wr[c] = int8_t(clamp(rint(v[j] * inv), -127.0f, 127.0f));
    if (tid == 0)
        sc[r] = ns;
}

struct LnAffineArgs {
    uint n;
    float eps;
};

// y = layer_norm(x) * (1 + scale) + shift over each row (no learned weights); one threadgroup per row.
kernel void ln_affine(device const float* x [[buffer(0)]], device const float* scale [[buffer(1)]],
                      device const float* shift [[buffer(2)]], device float* y [[buffer(3)]],
                      constant LnAffineArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += xr[i];
    const float mean = tg_sum(s, scratch, tid) / float(a.n);
    float v = 0.0f;
    for (uint i = tid; i < a.n; i += TG) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrt(tg_sum(v, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        yr[i] = (xr[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
}

// y = layer_norm(x) * w + b over each row; one threadgroup per row.
kernel void layer_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                       device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant LnAffineArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[TG / 32];
    device const float* xr = x + uint64_t(row) * a.n;
    device float* yr = y + uint64_t(row) * a.n;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += TG)
        s += xr[i];
    const float mean = tg_sum(s, scratch, tid) / float(a.n);
    float v = 0.0f;
    for (uint i = tid; i < a.n; i += TG) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float inv = rsqrt(tg_sum(v, scratch, tid) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += TG)
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// x += g * y over rows of n (g: [n]). Grid (n, rows).
kernel void scale_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                      device const float* g [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n)
        x[uint64_t(gid.y) * n + gid.x] += g[gid.x] * y[uint64_t(gid.y) * n + gid.x];
}

// Convolutions as GEMMs over channels-last images: the 3x3 unfold (stride 1 and 2), the group norm, GELU.

// Padding of the convolutions' unfold: 0 zeros, 1 the image reflected without its edge (PyTorch's reflect).
static int pad_coord(int i, int n, uint mode)
{
    if (mode == 1)
        return i < 0 ? -i : i >= n ? 2 * n - 2 - i : i;
    return i;
}

struct Im2colArgs {
    uint h, w, c;   // input image
    uint p0, rows;  // output pixels [p0, p0 + rows)
    uint kpad;      // columns, 9 * c padded to a multiple of 32
    uint mode;      // the padding's: 0 zeros, 1 reflect
};

// out[r][ch * 9 + ky * 3 + kx] = x[y + ky - 1][x + kx - 1][ch] for output pixel p0 + r = (y, x), zero outside the
// image and past 9 * c. Grid (kpad, rows).
kernel void im2col3x3(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                      constant Im2colArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x, r = gid.y;
    if (col >= a.kpad || r >= a.rows)
        return;
    float v = 0.0f;
    if (col < 9 * a.c) {
        const uint ch = col / 9, kk = col % 9, p = a.p0 + r;
        const int y = pad_coord(int(p / a.w) + int(kk / 3) - 1, int(a.h), a.mode);
        const int xx = pad_coord(int(p % a.w) + int(kk % 3) - 1, int(a.w), a.mode);
        if (y >= 0 && y < int(a.h) && xx >= 0 && xx < int(a.w))
            v = x[(uint64_t(y) * a.w + uint(xx)) * a.c + ch];
    }
    out[uint64_t(r) * a.kpad + col] = half(v);
}

// im2col3x3 for a stride-2 convolution (padding 1): output pixel p0 + r = (y, x) of an ow-wide image reads input
// pixels (2y + ky - 1, 2x + kx - 1) of the h x w input. Grid (kpad, rows).
struct Im2colS2Args {
    uint h, w, c, p0, rows, kpad, ow;
    uint pad;  // above and left of the image: 1, or 0 (then only below and right)
    uint mode; // the padding's: 0 zeros, 1 reflect
};
kernel void im2col3x3_s2(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                         constant Im2colS2Args& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x, r = gid.y;
    if (col >= a.kpad || r >= a.rows)
        return;
    float v = 0.0f;
    if (col < 9 * a.c) {
        const uint ch = col / 9, kk = col % 9, p = a.p0 + r;
        const int y = pad_coord(int(p / a.ow) * 2 + int(kk / 3) - int(a.pad), int(a.h), a.mode);
        const int xx = pad_coord(int(p % a.ow) * 2 + int(kk % 3) - int(a.pad), int(a.w), a.mode);
        if (y >= 0 && y < int(a.h) && xx >= 0 && xx < int(a.w))
            v = x[(uint64_t(y) * a.w + uint(xx)) * a.c + ch];
    }
    out[uint64_t(r) * a.kpad + col] = half(v);
}


// Group norm over channels-last images [rows][c]: per-chunk channel sums, then per-group statistics, then the norm.

struct GnArgs {
    uint rows, c, groups, chunk; // chunk: rows a threadgroup of gn_partial sums
    float eps;
    uint silu;
};

// part[k][ch] = (sum, sum of squares) of channel ch over rows [k * chunk, (k + 1) * chunk). Grid (c, chunks).
kernel void gn_partial(device const float* x [[buffer(0)]], device float2* part [[buffer(1)]],
                       constant GnArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x, k = gid.y;
    if (ch >= a.c)
        return;
    const uint r1 = min(a.rows, (k + 1) * a.chunk);
    float s = 0.0f, q = 0.0f;
    for (uint r = k * a.chunk; r < r1; ++r) {
        const float v = x[uint64_t(r) * a.c + ch];
        s += v;
        q += v * v;
    }
    part[uint64_t(k) * a.c + ch] = float2(s, q);
}

// stats[g] = (mean, 1 / sqrt(var + eps)) of group g from the chunks' sums. One thread per group.
kernel void gn_stats(device const float2* part [[buffer(0)]], device float2* stats [[buffer(1)]],
                     constant GnArgs& a [[buffer(2)]], uint g [[thread_position_in_grid]])
{
    if (g >= a.groups)
        return;
    const uint per = a.c / a.groups, chunks = (a.rows + a.chunk - 1) / a.chunk;
    float s = 0.0f, q = 0.0f;
    for (uint k = 0; k < chunks; ++k)
        for (uint ch = g * per; ch < (g + 1) * per; ++ch) {
            const float2 p = part[uint64_t(k) * a.c + ch];
            s += p.x;
            q += p.y;
        }
    const float n = float(a.rows) * float(per), mean = s / n;
    stats[g] = float2(mean, rsqrt(max(q / n - mean * mean, 0.0f) + a.eps));
}

// y = (x - mean) / sd * w + b per group, then SiLU when silu is set. Grid (c, rows).
kernel void gn_apply(device const float* x [[buffer(0)]], device const float2* stats [[buffer(1)]],
                     device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                     device float* y [[buffer(4)]], constant GnArgs& a [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    if (ch >= a.c || gid.y >= a.rows)
        return;
    const float2 st = stats[ch / (a.c / a.groups)];
    const uint64_t i = uint64_t(gid.y) * a.c + ch;
    float v = (x[i] - st.x) * st.y * w[ch] + b[ch];
    if (a.silu)
        v = v / (1.0f + exp(-v));
    y[i] = v;
}


// erf, max error ~1.2e-7 (Chebyshev fit of erfc).
static float erf_fit(float x)
{
    const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
    const float r =
        t * exp(-z * z - 1.26551223f +
                t * (1.00002368f +
                     t * (0.37409196f +
                          t * (0.09678418f +
                               t * (-0.18628806f +
                                    t * (0.27886807f +
                                         t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

// x = x / 2 * (1 + erf(x / sqrt 2)).
kernel void gelu_erf(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    if (i < n)
        x[i] = 0.5f * x[i] * (1.0f + erf_fit(x[i] * 0.70710678f));
}

// Block-sparse attention for video transformers (Video Sparse Attention's training-free form): the video rows sit
// in 64-row cubes (4 latent frames x 4 x 4 patches, padded rows marked invalid), each cube's queries attend the
// cubes a coarse pass picks (mean-pooled q and k per cube scored against each other, the top k per head) plus the
// dense rows (text, conditions, audio) after the cubes; the dense rows' queries attend everything.

struct CubeArgs {
    uint n_cubes, k_pad, dim; // cubes of 64 rows; rows of a head in the tiles pooled (q's and k's can differ); the width
    uint topk;                // cube_topk: cubes kept per query cube
    uint local;               // cube_topk: 1 keeps the query cube itself first, 3 also the cubes `stride` before and after
    uint stride;              // cubes a frame group apart (the temporal neighbours)
};

// out[h][c][d] = mean over the cube's valid rows of x[h][c * 64 + r][d] (cnt[c] valid rows). Grid (dim, cubes, heads).
kernel void cube_pool(device const half* x [[buffer(0)]], device const uint* cnt [[buffer(1)]],
                      device float* out [[buffer(2)]], constant CubeArgs& a [[buffer(3)]],
                      uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, c = gid.y, h = gid.z;
    if (d >= a.dim || c >= a.n_cubes)
        return;
    float s = 0.0f;
    device const half* p = x + (uint64_t(h) * a.k_pad + uint64_t(c) * 64) * a.dim + d;
    for (uint r = 0; r < 64; ++r)
        s += float(p[uint64_t(r) * a.dim]);
    out[(uint64_t(h) * a.n_cubes + c) * a.dim + d] = cnt[c] ? s / float(cnt[c]) : 0.0f;
}

// sc[h][qc][kc] = qp[h][qc] . kp[h][kc]. Grid (cubes (kc), cubes (qc), heads).
kernel void cube_scores(device const float* qp [[buffer(0)]], device const float* kp [[buffer(1)]],
                        device float* sc [[buffer(2)]], constant CubeArgs& a [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const uint kc = gid.x, qc = gid.y, h = gid.z;
    if (kc >= a.n_cubes || qc >= a.n_cubes)
        return;
    device const float* q = qp + (uint64_t(h) * a.n_cubes + qc) * a.dim;
    device const float* k = kp + (uint64_t(h) * a.n_cubes + kc) * a.dim;
    float s = 0.0f;
    for (uint d = 0; d < a.dim; d += 4)
        s += q[d] * k[d] + q[d + 1] * k[d + 1] + q[d + 2] * k[d + 2] + q[d + 3] * k[d + 3];
    sc[(uint64_t(h) * a.n_cubes + qc) * a.n_cubes + kc] = s;
}

// sel[h][qc][0 .. topk) = the key cubes with the largest scores for query cube qc (the largest first), nsel[h][qc] how
// many (fewer when there are fewer cubes). One threadgroup of 256 per (qc, h); at most 4096 cubes.
kernel void cube_topk(device const float* sc [[buffer(0)]], device ushort* sel [[buffer(1)]],
                      device ushort* nsel [[buffer(2)]], constant CubeArgs& a [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scores[4096];
    threadgroup float best_v[8];
    threadgroup uint best_i[8];
    const uint qc = tg.x, h = tg.y, n = a.n_cubes;
    device const float* row = sc + (uint64_t(h) * n + qc) * n;
    for (uint i = tid; i < n; i += 256)
        scores[i] = row[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k = min(a.topk, n);
    // The local cubes go in first, their scores taken out of the running.
    uint it = 0;
    if (a.local) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint picks[3] = {qc, qc >= a.stride ? qc - a.stride : n, qc + a.stride < n ? qc + a.stride : n};
            for (uint j = 0; j < a.local && it < k; ++j)
                if (picks[j] < n) {
                    sel[(uint64_t(h) * n + qc) * a.topk + it++] = ushort(picks[j]);
                    scores[picks[j]] = -INFINITY;
                }
            best_i[0] = it;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        it = best_i[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (; it < k; ++it) {
        float v = -INFINITY;
        uint idx = 0;
        for (uint i = tid; i < n; i += 256)
            if (scores[i] > v) {
                v = scores[i];
                idx = i;
            }
        // The simdgroup's best, then the threadgroup's: the larger score, the smaller index on ties.
        for (uint o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(v, o);
            const uint oi = simd_shuffle_xor(idx, o);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) {
            best_v[sg] = v;
            best_i[sg] = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bv = best_v[0];
            uint bi = best_i[0];
            for (uint j = 1; j < 8; ++j)
                if (best_v[j] > bv || (best_v[j] == bv && best_i[j] < bi)) {
                    bv = best_v[j];
                    bi = best_i[j];
                }
            sel[(uint64_t(h) * n + qc) * a.topk + it] = ushort(bi);
            scores[bi] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0)
        nsel[uint64_t(h) * n + qc] = ushort(k);
}

struct SparseAttnArgs {
    uint n_q, q_pad, n_k, k_pad, heads, kv_heads, width; // as AttnArgs (rows in the tiles' order)
    float scale;
    uint n_cubes, max_sel;  // the first n_cubes query tiles are cubes; sel rows hold max_sel entries
    uint dense0, dense1;    // the dense rows [dense0, dense1), attended by every cube's queries
};

// flash_attn_coop over a pass list: query tile qt < n_cubes visits its selected key cubes (two passes each) and the
// dense rows; a later tile visits every key. Keys with valid[j] == 0 (padding) are masked. The output goes to row
// map[q] of o (the tiles' order mapped back to the model's; rows with map[q] < 0 are dropped).
kernel void flash_attn_sparse_128(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                                  device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                                  device const ushort* sel [[buffer(4)]], device const ushort* nsel [[buffer(5)]],
                                  device const uchar* valid [[buffer(6)]], device const int* map [[buffer(7)]],
                                  constant SparseAttnArgs& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed)
    constexpr int D = 128, R = 64, NSG = 4, K = 32, CAP = R * K / 128, ACAP = R * D / 128, CQ = K / 32, AQ = D / 32;
    threadgroup half p_all[2][R * K];
    threadgroup float pmax[2][2][R];
    threadgroup float lsum[2][R];
    const uint qt = tg.x, q0 = qt * R, h = tg.y, kvh = h / (a.heads / a.kv_heads);
    if (q0 >= a.n_q)
        return;
    const uint64_t qhead = uint64_t(h) * a.q_pad * D, khead = uint64_t(kvh) * a.k_pad * D;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qhead + uint64_t(q0) * D,
                                                                dextents<int32_t, 2>(D, R));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(const_cast<device half*>(v) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp0(p_all[0], dextents<int32_t, 2>(K, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp1(p_all[1], dextents<int32_t, 2>(K, R));
    constexpr auto qk_desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(R, D, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp0), decltype(vt0), float>();
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] = 0.0f;
    const uint sgx = sg & 1, sgy = sg >> 1;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2) | (sgy << 4);
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1)) + 16 * sgx;
    const bool writer = (lane & 9) == 0;
    const float scale = a.scale * 1.4426950408889634f;
    // The passes: a cube's selected key cubes, two of 32 keys each, then the dense rows; a dense tile's all keys.
    const bool cube = qt < a.n_cubes;
    const uint ns = cube ? uint(nsel[uint64_t(h) * a.n_cubes + qt]) : 0;
    device const ushort* my_sel = sel + (uint64_t(h) * a.n_cubes + qt) * a.max_sel;
    const uint dense_passes = cube ? (a.dense1 - a.dense0 + K - 1) / K : (a.n_k + K - 1) / K;
    const uint n_pass = 2 * ns + dense_passes;
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint buf = 0;
    for (uint p = 0; p < n_pass; ++p) {
        uint j0, hi;
        if (p < 2 * ns) {
            j0 = uint(my_sel[p >> 1]) * 2 * K + (p & 1) * K;
            hi = j0 + K;
        } else if (cube) {
            j0 = a.dense0 + (p - 2 * ns) * K;
            hi = min(j0 + K, a.dense1);
        } else {
            j0 = p * K;
            hi = min(j0 + K, a.n_k);
        }
        for (uint16_t i = 0; i < CAP; ++i)
            sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        // Every pass masked by range and validity (a flag for all-valid passes that skipped this measured no faster).
        float mt[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * CQ)) << 1);
            const uint j = j0 + ((i >> 3) % CQ) * 32 + c_base + (i & 3);
            float x = sc[i] * scale;
            if (j >= hi || valid[j] == 0)
                x = -INFINITY;
            sc[i] = x;
            mt[slot] = max(mt[slot], x);
        }
        for (int s = 0; s < 4; ++s) {
            mt[s] = max(mt[s], simd_shuffle_xor(mt[s], 1));
            mt[s] = max(mt[s], simd_shuffle_xor(mt[s], 8));
        }
        if (writer)
            for (int s = 0; s < 4; ++s)
                pmax[buf][sgx][r0 + (s & 1) * 8 + (s >> 1) * 32] = mt[s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool rescale = false;
        float f[4];
        for (int s = 0; s < 4; ++s) {
            mt[s] = max(mt[s], pmax[buf][sgx ^ 1][r0 + (s & 1) * 8 + (s >> 1) * 32]);
            f[s] = 1.0f;
            if (mt[s] > m[s] + 8.0f) {
                f[s] = m[s] == -INFINITY ? 0.0f : exp2(m[s] - mt[s]);
                m[s] = mt[s];
                l[s] *= f[s];
                rescale = true;
            }
        }
        if (simd_any(rescale))
            for (uint16_t i = 0; i < ACAP; ++i)
                acc[i] *= f[((i >> 2) & 1) | ((i / (8 * AQ)) << 1)];
        threadgroup half* pp = p_all[buf];
        for (uint16_t i = 0; i < CAP; i += 4) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * CQ)) << 1);
            const uint row = r0 + (slot & 1) * 8 + (slot >> 1) * 32, col = ((i >> 3) % CQ) * 32 + c_base;
            const float mm = m[slot];
            half4 e;
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                const float ex = mm == -INFINITY ? 0.0f : exp2(sc[i + c] - mm);
                e[c] = half(ex);
                sum += ex;
            }
            *reinterpret_cast<threadgroup half4*>(pp + row * K + col) = e;
            l[slot] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto vt = tv.slice(0, int(j0));
        if (buf == 0)
            pv.run(tp0, vt, acc);
        else
            pv.run(tp1, vt, acc);
        buf ^= 1;
    }
    for (int s = 0; s < 4; ++s) {
        l[s] += simd_shuffle_xor(l[s], 1);
        l[s] += simd_shuffle_xor(l[s], 8);
    }
    if (writer)
        for (int s = 0; s < 4; ++s)
            lsum[sgx][r0 + (s & 1) * 8 + (s >> 1) * 32] = l[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[4];
    for (int s = 0; s < 4; ++s) {
        const float total = l[s] + lsum[sgx ^ 1][r0 + (s & 1) * 8 + (s >> 1) * 32];
        inv[s] = total > 0.0f ? 1.0f / total : 0.0f;
    }
    // Each lane's rows are r0 + {0, 8, 32, 40} of the tile; element i's column ((i >> 3) % AQ) * 32 + c_base + (i & 3).
    for (uint16_t i = 0; i < ACAP; ++i) {
        const uint slot = ((i >> 2) & 1) | ((i / (8 * AQ)) << 1);
        const uint row = q0 + r0 + (slot & 1) * 8 + (slot >> 1) * 32;
        if (row >= a.n_q)
            continue;
        const int dst = map[row];
        if (dst < 0)
            continue;
        const uint col = ((i >> 3) % AQ) * 32 + c_base + (i & 3);
        o[uint64_t(dst) * a.width + h * D + col] = acc[i] * inv[slot];
    }
}
