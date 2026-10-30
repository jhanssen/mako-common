// Generic transformer pieces any model can use: attention heads in half tiles (with rotary positions), masked flash
// attention with grouped key/value heads, activations. Activations are fp32 rows [token][width] unless noted.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

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

// o[q][h * D + d] = sum_j softmax_j(scale * q . k_j) v_j[d] over the keys j query q may see: j <= its position, or
// both at or past `block`. q half [heads][q_pad][D], k and v half [kv_heads][k_pad][D] (head h reads kv head
// h / (heads / kv_heads)). A threadgroup takes B queries; the scores of K keys at a time go to threadgroup memory for
// the running softmax (256 / B threads a row, in base 2), the output stays in registers and both products run through
// matmul2d with float accumulation. A row's reference maximum moves only when a score passes it by 8 (the
// exponentials stay within 256, half's range), so most passes leave the output unscaled. q_pad a multiple of B,
// k_pad of K. Grid (q_pad / B, heads), 8 simdgroups. Measured at 32 heads, 4096 queries x 4122 keys: 32 x 128 13.5 ->
// 12.2 ms with the lazy maximum; 64 x 64 14.5, 128 x 32 29 ms (the score store and two barriers a pass, not the key
// and value reads, set the pace).
template <int D, int B, int K>
kernel void flash_attn(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                       constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp math_mode(relaxed) // INF / NaN are values here (the default fast math assumes none: MSL 1.6.3)
    constexpr int TPR = 256 / B, CPT = K / TPR; // threads a row, columns a thread
    threadgroup float s[B * K];
    threadgroup half p[B * K];
    threadgroup float row_scale[B];
    threadgroup uint rescale;
    if (tid == 0)
        rescale = 0;
    const uint q0 = tg.x * B, h = tg.y, kvh = h / (a.heads / a.kv_heads);
    const uint64_t qhead = uint64_t(h) * a.q_pad * D, khead = uint64_t(kvh) * a.k_pad * D;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qhead + uint64_t(q0) * D,
                                                                dextents<int32_t, 2>(D, B));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tv(const_cast<device half*>(v) + khead,
                                                                dextents<int32_t, 2>(D, int(a.k_pad)));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> ts(s, dextents<int32_t, 2>(K, B));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(p, dextents<int32_t, 2>(K, B));
    constexpr auto qk_desc = matmul2d_descriptor(B, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(B, D, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<8>> qk;
    matmul2d<pv_desc, execution_simdgroups<8>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp), decltype(vt0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0.0f;
    // Each thread keeps its row's running maximum and sum.
    const uint r = tid / TPR, c0 = (tid % TPR) * CPT;
    const uint pos = a.q_pos0 + q0 + r;
    // Keys past the block's last query position, when no query of it reaches the image block, are never seen.
    const uint last = a.q_pos0 + min(q0 + B, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f; // to base 2
    float m = -INFINITY, l = 0.0f;
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < sc.get_capacity(); ++i)
            if (sc.is_valid_element(i))
                sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        sc.store(ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
            rescale = 1;
        }
        float sum = 0.0f;
        for (uint c = 0; c < CPT; ++c) {
            const float e = m == -INFINITY ? 0.0f : exp2(x[c] - m);
            p[r * K + c0 + c] = half(e);
            sum += e;
        }
        for (ushort mm = 1; mm < TPR; mm *= 2)
            sum += simd_shuffle_xor(sum, mm);
        l = l * f + sum;
        if (tid % TPR == 0)
            row_scale[r] = f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale)
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                if (acc.is_valid_element(i))
                    acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
        auto vt = tv.slice(0, int(j0));
        pv.run(tp, vt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0)
            rescale = 0;
    }
    if (tid % TPR == 0)
        row_scale[r] = l > 0.0f ? 1.0f / l : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] *= row_scale[acc.get_multidimensional_index(i)[1]];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}

#define FLASH_ATTN(d, b, k)                                                                                           \
    template [[host_name("flash_attn_" #d "_" #b "x" #k)]] kernel void flash_attn<d, b, k>(                           \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort);
FLASH_ATTN(128, 32, 128)

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
