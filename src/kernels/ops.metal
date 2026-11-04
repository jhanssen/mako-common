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
FLASH_ATTN_SG(256, 16, 32, 4)
FLASH_ATTN_SG(64, 16, 32, 4)

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
};

// y[h][t][d] = half of head h of row t, RMS-normed (times w) when eps is set, its first rot values rotated by
// cs[t][i] = (cos, sin). Threadgroups (rows, heads) of dim threads.
kernel void heads_partial(device const float* x [[buffer(0)]], device const float2* cs [[buffer(1)]],
                          device half* y [[buffer(2)]], device const float* w [[buffer(3)]],
                          constant HeadsPartialArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint i [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float vals[256];
    threadgroup float part[8];
    const uint t = tg.x, h = tg.y;
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
    y[(uint64_t(h) * a.n_pad + t) * a.dim + i] = half(v);
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
