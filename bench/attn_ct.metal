// Experiments behind ops.metal's flash_attn_coop (kernels named xp_flash_attn_...; attn_bench runs them by name,
// coop_probe / coop_probe4 print the lane layouts, mm_rate_bench the matmul rates). 56 heads of 128 at 19308 keys,
// against 336-340 ms for flash_attn_sg_128_16x64_4:
//   flash_attn_ct: one simdgroup on 16 rows with the softmax on the cooperative tensor, no score buffer: 348-407 ms
//     (128 keys a pass spills; mm_rate_bench: 16 x 64 matmul2d tiles run at the 60-65 TFLOP/s of 64-row ones once the
//     clocks are up, 16 x 128 at 39).
//   flash_attn_coop: 4 simdgroups on 64 rows, 32 keys a pass: 297-304 ms, the kernel adopted (each key and value
//     row read by two simdgroups instead of four); 64 keys spill (427-485).
//     _sameblock reads key block 0 every pass: 207 ms, the compute-only time; _noexchange skips the maximum
//     exchange: 303 ms, so the barriers cost nothing measurable and the K / V streaming a third.
//   flash_attn_coop_pair: two 32-key products a pass, one exchange per 64 keys: 457 ms.
//   flash_attn_coop8: 8 simdgroups on 64 rows, 64 or 128 keys a pass: 327 / 363 ms.
//   flash_attn_coop_tg: K and V staged in threadgroup memory, K a pass ahead: 285 ms at 19k but 1324 against 1212
//     at 38k.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Which (column, row) of a 16 x K result each lane's cooperative-tensor elements are, for the Q K^T product
// (R rows x K keys from D-deep operands). out[lane][i] = (valid, col, row); out[32 * CAP * 3] = capacity.
template <int R, int K, int D>
kernel void coop_layout_probe(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]])
{
    threadgroup half a[R * D];
    threadgroup half b[K * D];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(D, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tb(b, dextents<int32_t, 2>(D, K));
    constexpr auto desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto sc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), float>();
    const uint cap = sc.get_capacity();
    for (uint16_t i = 0; i < cap; ++i) {
        const auto idx = sc.get_multidimensional_index(i);
        out[(lane * cap + i) * 3] = sc.is_valid_element(i);
        out[(lane * cap + i) * 3 + 1] = idx[0];
        out[(lane * cap + i) * 3 + 2] = idx[1];
    }
    if (lane == 0)
        out[32 * 4096 * 3] = int(cap);
}
template [[host_name("coop_layout_probe_16x64")]] kernel void coop_layout_probe<16, 64, 128>(device int*, uint);
template [[host_name("coop_layout_probe_16x128")]] kernel void coop_layout_probe<16, 128, 128>(device int*, uint);
// The P V product's destination: R rows x D columns from K-deep operands.
template [[host_name("coop_layout_probe_16x128_pv")]] kernel void coop_layout_probe<16, 128, 64>(device int*, uint);

struct AttnArgs {
    uint n_q, q_pad, n_k, k_pad, heads, kv_heads, q_pos0, block, width;
    float scale;
};

// Flash attention with the softmax on the cooperative tensor: for a 16 x N result tile, lane l holds rows
// r0 = ((l >> 1) & 3) | ((l >> 4) << 2) and r0 + 8, and in each 16-column block the 4 columns from
// 4 ((l & 1) + 2 ((l >> 3) & 1)); element i is block i / 8, row (i / 4) & 1, column i & 3. The four lanes of a row
// differ in bits 0 and 3. So a row's maximum and sum take two shuffles, the scores never leave the registers, and
// only P (half) goes through threadgroup memory: 128 keys a pass fit 4 simdgroups.
template <int D, int K, int NSG, int MODE>
kernel void flash_attn_ct(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                          device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                          constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed)
    constexpr int R = 16, CAP = K / 2;
    threadgroup half p_all[NSG][R * K];
    threadgroup half* p = p_all[sg];
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
        acc[i] = 0.0f;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2), r1 = r0 + 8;
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1));
    const uint pos0 = a.q_pos0 + q0 + r0, pos1 = pos0 + 8;
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < CAP; ++i)
            sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        if constexpr (MODE != 3)
            qk.run(tq, kt, sc);
        if constexpr (MODE == 1 || MODE == 3) {
            for (uint16_t i = 0; i < CAP; ++i) {
                const uint col = (i >> 3) * 16 + c_base + (i & 3);
                const bool second = (i >> 2) & 1;
                p[(second ? r1 : r0) * K + col] = half(MODE == 3 ? 0.01f : sc[i] * 0.001f);
            }
            l0 = l1 = 1.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto vt = tv.slice(0, int(j0));
            pv.run(tp, vt, acc);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        float mt0 = -INFINITY, mt1 = -INFINITY;
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint j = j0 + (i >> 3) * 16 + c_base + (i & 3);
            const bool second = (i >> 2) & 1;
            const uint pos = second ? pos1 : pos0;
            const bool seen = j < a.n_k && (j <= pos || (pos >= a.block && j >= a.block));
            const float x = seen ? sc[i] * scale : -INFINITY;
            sc[i] = x;
            if (second)
                mt1 = max(mt1, x);
            else
                mt0 = max(mt0, x);
        }
        mt0 = max(mt0, simd_shuffle_xor(mt0, 1));
        mt0 = max(mt0, simd_shuffle_xor(mt0, 8));
        mt1 = max(mt1, simd_shuffle_xor(mt1, 1));
        mt1 = max(mt1, simd_shuffle_xor(mt1, 8));
        // The reference maximum moves only when a score passes it by 8 (the exponentials stay within 256, half's
        // range), so most passes leave the output unscaled.
        float f0 = 1.0f, f1 = 1.0f;
        if (mt0 > m0 + 8.0f) {
            f0 = m0 == -INFINITY ? 0.0f : exp2(m0 - mt0);
            m0 = mt0;
        }
        if (mt1 > m1 + 8.0f) {
            f1 = m1 == -INFINITY ? 0.0f : exp2(m1 - mt1);
            m1 = mt1;
        }
        float s0 = 0.0f, s1 = 0.0f;
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint col = (i >> 3) * 16 + c_base + (i & 3);
            const bool second = (i >> 2) & 1;
            const float m = second ? m1 : m0;
            const float e = m == -INFINITY ? 0.0f : exp2(sc[i] - m);
            p[(second ? r1 : r0) * K + col] = half(e);
            if (second)
                s1 += e;
            else
                s0 += e;
        }
        s0 += simd_shuffle_xor(s0, 1);
        s0 += simd_shuffle_xor(s0, 8);
        s1 += simd_shuffle_xor(s1, 1);
        s1 += simd_shuffle_xor(s1, 8);
        l0 = l0 * f0 + s0;
        l1 = l1 * f1 + s1;
        if (simd_any(f0 != 1.0f || f1 != 1.0f))
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                acc[i] *= ((i >> 2) & 1) ? f1 : f0;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (MODE != 2) {
            auto vt = tv.slice(0, int(j0));
            pv.run(tp, vt, acc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        acc[i] *= ((i >> 2) & 1) ? inv1 : inv0;
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}

#define FLASH_ATTN_CT(d, k, nsg, mode, suffix)                                                                        \
    template [[host_name("xp_flash_attn_ct_" #d "_16x" #k "_" #nsg suffix)]] kernel void flash_attn_ct<d, k, nsg, mode>( \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, \
        ushort);
FLASH_ATTN_CT(128, 64, 4, 0, "")
FLASH_ATTN_CT(128, 64, 4, 1, "_nosoftmax")
FLASH_ATTN_CT(128, 64, 4, 2, "_nopv")
FLASH_ATTN_CT(128, 64, 4, 3, "_noqk")
FLASH_ATTN_CT(128, 32, 4, 0, "")
FLASH_ATTN_CT(128, 32, 4, 1, "_nosoftmax")

// matmul2d's rate on attention-sized tiles: `iters` products of a fixed left tile with successive key tiles.
// Threadgroups of NSG simdgroups; cooperative ones run one op over all NSG, independent ones one op a simdgroup.
struct RateArgs {
    uint iters, k_pad;
};
template <int M, int N, int Dp, int NSG, bool COOP>
kernel void mm_rate(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                    device float* o [[buffer(2)]], constant RateArgs& a [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]])
{
    constexpr int ROWS = COOP ? M : M; // rows of this op's tile (the same shape either way)
    const uint64_t qoff = COOP ? uint64_t(tg.x) * M * Dp : (uint64_t(tg.x) * NSG + sg) * M * Dp;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qoff,
                                                                dextents<int32_t, 2>(Dp, ROWS));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device half*>(k),
                                                                dextents<int32_t, 2>(Dp, int(a.k_pad)));
    constexpr auto desc = matmul2d_descriptor(ROWS, N, Dp, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<COOP ? NSG : 1>> op;
    auto kt0 = tk.slice(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        acc[i] = 0.0f;
    for (uint it = 0; it < a.iters; ++it) {
        auto kt = tk.slice(0, int((it * N) % a.k_pad));
        op.run(tq, kt, acc);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + qoff / Dp * N, dextents<int32_t, 2>(N, ROWS));
    acc.store(to.slice(0, 0));
}
#define MM_RATE(m, n, d, nsg, coop, name)                                                                            \
    template [[host_name(name)]] kernel void mm_rate<m, n, d, nsg, coop>(device const half*, device const half*,      \
                                                                        device float*, constant RateArgs&, uint2, ushort);
MM_RATE(16, 64, 128, 4, false, "mm_rate_16x64x128_sg")
MM_RATE(16, 128, 128, 4, false, "mm_rate_16x128x128_sg")
MM_RATE(32, 64, 128, 4, false, "mm_rate_32x64x128_sg")
MM_RATE(64, 64, 128, 4, true, "mm_rate_64x64x128_coop4")
MM_RATE(64, 128, 128, 4, true, "mm_rate_64x128x128_coop4")
MM_RATE(32, 64, 128, 2, true, "mm_rate_32x64x128_coop2")

// The same probe for a cooperative product over NSG simdgroups: out[(sg * 32 + lane) * CAP + i] = (valid, col, row).
// T, ACC: the operands' and the result's element types.
template <int R, int N, int D, int NSG, typename T = half, typename ACC = float>
kernel void coop_layout_probe_coop(device int* out [[buffer(0)]], ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup T a[R * D];
    threadgroup T b[N * D];
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(D, R));
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> tb(b, dextents<int32_t, 2>(D, N));
    constexpr auto desc = matmul2d_descriptor(R, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto sc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), ACC>();
    const uint cap = sc.get_capacity();
    for (uint16_t i = 0; i < cap; ++i) {
        const auto idx = sc.get_multidimensional_index(i);
        out[((sg * 32 + lane) * cap + i) * 3] = sc.is_valid_element(i);
        out[((sg * 32 + lane) * cap + i) * 3 + 1] = idx[0];
        out[((sg * 32 + lane) * cap + i) * 3 + 2] = idx[1];
    }
    if (sg == 0 && lane == 0)
        out[128 * 4096 * 3] = int(cap);
}
template [[host_name("coop_layout_probe_coop4_64x64")]] kernel void coop_layout_probe_coop<64, 64, 128, 4>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop2_32x64")]] kernel void coop_layout_probe_coop<32, 64, 128, 2>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop2_32x128")]] kernel void coop_layout_probe_coop<32, 128, 64, 2>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop4_64x128")]] kernel void coop_layout_probe_coop<64, 128, 64, 4>(device int*, ushort, ushort);

// Flash attention with the four simdgroups of a threadgroup cooperating on 64 query rows: Q K^T and P V are one
// cooperative matmul2d each over 64-row tiles, so every key and value row is read by two simdgroups instead of four.
// A cooperative 64 x N result is a 2 x 2 grid
// of 32 x N/2 blocks over the simdgroups (sg >> 1 picks the 16-row sub-block of each 32-row half, sg & 1 the 16-column
// sub-block of each 32-column quarter); within a block lane l holds the single-simdgroup pattern (rows r0, r0 + 8,
// r0 + 32, r0 + 40; columns c_base .. c_base + 3 of each 32). A row's scores sit in two simdgroups, so its maximum is
// exchanged through threadgroup memory once a pass; its sum is kept per simdgroup and added at the end. P is
// double-buffered: two threadgroup barriers a pass.
template <int D, int K, bool SAME_BLOCK, bool NO_EXCHANGE = false>
kernel void flash_attn_coop(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                            device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                            constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed)
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
        auto kt = tk.slice(0, SAME_BLOCK ? 0 : int(j0));
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
        if (!NO_EXCHANGE) {
            if (writer)
                for (int s = 0; s < 4; ++s)
                    pmax[buf][sgx][r0 + (s & 1) * 8 + (s >> 1) * 32] = mt[s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // The reference maximum moves only when a score passes it by 8 (the exponentials stay within 256, half's
        // range), so most passes leave the output unscaled.
        bool rescale = false;
        float f[4];
        for (int s = 0; s < 4; ++s) {
            if (!NO_EXCHANGE)
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
        auto vt = tv.slice(0, SAME_BLOCK ? 0 : int(j0));
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
#define FLASH_ATTN_COOP(d, k, same, suffix)                                                                          \
    template [[host_name("xp_flash_attn_coop_" #d "_" #k suffix)]] kernel void flash_attn_coop<d, k, same>(              \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, \
        ushort);
FLASH_ATTN_COOP(128, 64, false, "")
FLASH_ATTN_COOP(128, 32, false, "")
FLASH_ATTN_COOP(128, 32, true, "_sameblock")
FLASH_ATTN_COOP(64, 32, false, "")
template [[host_name("xp_flash_attn_coop_128_32_noexchange")]] kernel void flash_attn_coop<128, 32, false, true>(
    device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, ushort);
template [[host_name("coop_layout_probe_coop8_128x32")]] kernel void coop_layout_probe_coop<128, 32, 128, 8>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop8_64x64")]] kernel void coop_layout_probe_coop<64, 64, 128, 8>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop8_64x128")]] kernel void coop_layout_probe_coop<64, 128, 64, 8>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_coop8_128x128")]] kernel void coop_layout_probe_coop<128, 128, 32, 8>(device int*, ushort, ushort);
template [[host_name("coop_layout_probe_i8_coop4_64x128")]] kernel void coop_layout_probe_coop<64, 128, 128, 4, int8_t, int32_t>(device int*, ushort, ushort);

// The cooperative kernel with two 32-key products a pass: one maximum exchange and one P barrier per 64 keys, the
// scores of 32 keys a cooperative tensor (a 64-key one spilled).
template <int D>
kernel void flash_attn_coop_pair(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                                 device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                                 constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed)
    constexpr int R = 64, NSG = 4, K = 32, CAP = R * K / 128, ACAP = R * D / 128, AQ = D / 32;
    threadgroup half p_all[2][R * 2 * K];
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
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp0(p_all[0], dextents<int32_t, 2>(2 * K, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp1(p_all[1], dextents<int32_t, 2>(2 * K, R));
    constexpr auto qk_desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(R, D, 2 * K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv;
    auto kt0 = tk.slice(0, 0);
    auto sa = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto sb = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), float>();
    auto vt0 = tv.slice(0, 0);
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp0), decltype(vt0), float>();
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] = 0.0f;
    const uint sgx = sg & 1, sgy = sg >> 1;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2) | (sgy << 4);
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1)) + 16 * sgx;
    const bool writer = (lane & 9) == 0;
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint buf = 0;
    // Element i of a 64 x 32 score tensor: row slot ((i >> 2) & 1) | ((i >> 3) << 1), column c_base + (i & 3).
    for (uint j0 = 0; j0 < n_keys; j0 += 2 * K) {
        for (uint16_t i = 0; i < CAP; ++i)
            sa[i] = sb[i] = 0.0f;
        auto kta = tk.slice(0, int(j0));
        qk.run(tq, kta, sa);
        auto ktb = tk.slice(0, int(j0 + K));
        qk.run(tq, ktb, sb);
        const bool plain = j0 + 2 * K <= a.n_k && ((a.q_pos0 + q0 >= a.block && j0 >= a.block) ||
                                                   j0 + 2 * K - 1 <= a.q_pos0 + q0);
        float mt[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint slot = ((i >> 2) & 1) | ((i >> 3) << 1);
            float xa = sa[i] * scale, xb = sb[i] * scale;
            if (!plain) {
                const uint j = j0 + c_base + (i & 3);
                const uint pos = a.q_pos0 + q0 + r0 + (slot & 1) * 8 + (slot >> 1) * 32;
                if (!(j < a.n_k && (j <= pos || (pos >= a.block && j >= a.block))))
                    xa = -INFINITY;
                if (!(j + K < a.n_k && (j + K <= pos || (pos >= a.block && j + K >= a.block))))
                    xb = -INFINITY;
            }
            sa[i] = xa;
            sb[i] = xb;
            mt[slot] = max(mt[slot], max(xa, xb));
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
        threadgroup half* p = p_all[buf];
        for (uint16_t i = 0; i < CAP; i += 4) {
            const uint slot = ((i >> 2) & 1) | ((i >> 3) << 1);
            const uint row = r0 + (slot & 1) * 8 + (slot >> 1) * 32;
            const float mm = m[slot];
            half4 ea, eb;
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                const float xa = mm == -INFINITY ? 0.0f : exp2(sa[i + c] - mm);
                const float xb = mm == -INFINITY ? 0.0f : exp2(sb[i + c] - mm);
                ea[c] = half(xa);
                eb[c] = half(xb);
                sum += xa + xb;
            }
            *reinterpret_cast<threadgroup half4*>(p + row * 2 * K + c_base) = ea;
            *reinterpret_cast<threadgroup half4*>(p + row * 2 * K + K + c_base) = eb;
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
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] *= inv[((i >> 2) & 1) | ((i / (8 * AQ)) << 1)];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}
template [[host_name("xp_flash_attn_coop_pair_128")]] kernel void flash_attn_coop_pair<128>(
    device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, ushort);

// Eight simdgroups on 64 query rows: the scores of K keys split 4 ways over the columns (16 a simdgroup) and 2 ways
// over the rows, the output 2 x 4 over 32-column blocks; every lane holds 4 rows (r0, +8, +32, +40). A row's maximum
// gathers 4 partials through threadgroup memory. 64 keys a pass with P double-buffered (two barriers a pass); 128
// keys with one P buffer (three).
template <int D, int K>
kernel void flash_attn_coop8(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                             device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                             constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]])
{
#pragma METAL fp math_mode(relaxed)
    constexpr int R = 64, NSG = 8, CAP = R * K / 256, ACAP = R * D / 256, CB = K / 16; // column blocks of the scores
    constexpr int BUFS = K <= 64 ? 2 : 1;
    threadgroup half p_all[BUFS][R * K];
    threadgroup float pmax[2][4][R];
    threadgroup float lsum[4][R];
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
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp1(p_all[BUFS - 1], dextents<int32_t, 2>(K, R));
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
    const uint sgx = sg & 3, sgy = sg >> 2;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2) | (sgy << 4);
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1));
    const bool writer = (lane & 9) == 0;
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint buf = 0;
    // Score element i: row slot ((i >> 2) & 1) | ((i / (8 * CB / 4)) << 1)... with CB column blocks of 16 over
    // 4 simdgroups, a simdgroup holds CB / 4 blocks of 16: element i is block (i >> 3) % (CB / 4), row half
    // i / (8 * CB / 4), column 16 (sgx + 4 block) + c_base + (i & 3).
    constexpr int SB = CB / 4; // 16-column blocks a simdgroup
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < CAP; ++i)
            sc[i] = 0.0f;
        auto kt = tk.slice(0, int(j0));
        qk.run(tq, kt, sc);
        const bool plain = j0 + K <= a.n_k && ((a.q_pos0 + q0 >= a.block && j0 >= a.block) ||
                                               j0 + K - 1 <= a.q_pos0 + q0);
        float mt[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint16_t i = 0; i < CAP; ++i) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * SB)) << 1);
            float x = sc[i] * scale;
            if (!plain) {
                const uint j = j0 + 16 * (sgx + 4 * ((i >> 3) % SB)) + c_base + (i & 3);
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
        bool rescale = false;
        float f[4];
        for (int s = 0; s < 4; ++s) {
            const uint row = r0 + (s & 1) * 8 + (s >> 1) * 32;
            mt[s] = max(max(mt[s], pmax[buf][sgx ^ 1][row]), max(pmax[buf][sgx ^ 2][row], pmax[buf][sgx ^ 3][row]));
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
                acc[i] *= f[((i >> 2) & 1) | ((i >> 4) << 1)];
        threadgroup half* p = p_all[buf];
        for (uint16_t i = 0; i < CAP; i += 4) {
            const uint slot = ((i >> 2) & 1) | ((i / (8 * SB)) << 1);
            const uint row = r0 + (slot & 1) * 8 + (slot >> 1) * 32;
            const uint col = 16 * (sgx + 4 * ((i >> 3) % SB)) + c_base;
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
        if constexpr (BUFS == 1)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        else
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
        const uint row = r0 + (s & 1) * 8 + (s >> 1) * 32;
        const float total = lsum[0][row] + lsum[1][row] + lsum[2][row] + lsum[3][row];
        inv[s] = total > 0.0f ? 1.0f / total : 0.0f;
    }
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] *= inv[((i >> 2) & 1) | ((i >> 4) << 1)];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}
#define FLASH_ATTN_COOP8(d, k)                                                                                       \
    template [[host_name("xp_flash_attn_coop8_" #d "_" #k)]] kernel void flash_attn_coop8<d, k>(                         \
        device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, \
        ushort);
FLASH_ATTN_COOP8(128, 64)
FLASH_ATTN_COOP8(128, 128)

// The four-simdgroup kernel with each pass's K and V tiles staged in threadgroup memory by one coalesced read a
// threadgroup (the cooperative products read every key twice otherwise). K is loaded a pass ahead, between the two
// barriers; V for the pass in the same place. K = 32 keys a pass: K 8 KB, V 8 KB, P 2 x 4 KB.
template <int D, int K>
kernel void flash_attn_coop_tg(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                               device const half* v [[buffer(2)]], device float* o [[buffer(3)]],
                               constant AttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                               ushort tid [[thread_index_in_threadgroup]])
{
#pragma METAL fp math_mode(relaxed)
    constexpr int R = 64, NSG = 4, CAP = R * K / 128, ACAP = R * D / 128, CQ = K / 32, AQ = D / 32;
    constexpr int TILE16 = K * D / 8, PER_THREAD = TILE16 / (32 * NSG); // 16-byte pieces of a tile, a thread's share
    threadgroup half kbuf[K * D];
    threadgroup half vbuf[K * D];
    threadgroup half p_all[2][R * K];
    threadgroup float pmax[2][2][R];
    threadgroup float lsum[2][R];
    const uint q0 = tg.x * R, h = tg.y, kvh = h / (a.heads / a.kv_heads);
    if (q0 >= a.n_q)
        return;
    const uint64_t qhead = uint64_t(h) * a.q_pad * D, khead = uint64_t(kvh) * a.k_pad * D;
    device const uint4* kv8 = reinterpret_cast<device const uint4*>(k + khead);
    device const uint4* vv8 = reinterpret_cast<device const uint4*>(v + khead);
    threadgroup uint4* kb8 = reinterpret_cast<threadgroup uint4*>(kbuf);
    threadgroup uint4* vb8 = reinterpret_cast<threadgroup uint4*>(vbuf);
    auto load_k = [&](uint j0) {
        for (int i = 0; i < PER_THREAD; ++i)
            kb8[tid + i * 32 * NSG] = kv8[uint64_t(j0) * D / 8 + tid + i * 32 * NSG];
    };
    auto load_v = [&](uint j0) {
        for (int i = 0; i < PER_THREAD; ++i)
            vb8[tid + i * 32 * NSG] = vv8[uint64_t(j0) * D / 8 + tid + i * 32 * NSG];
    };
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device half*>(q) + qhead + uint64_t(q0) * D,
                                                                dextents<int32_t, 2>(D, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tk(kbuf, dextents<int32_t, 2>(D, K));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tv(vbuf, dextents<int32_t, 2>(D, K));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp0(p_all[0], dextents<int32_t, 2>(K, R));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp1(p_all[1], dextents<int32_t, 2>(K, R));
    constexpr auto qk_desc = matmul2d_descriptor(R, K, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc =
        matmul2d_descriptor(R, D, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv;
    auto sc = qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>();
    auto acc = pv.template get_destination_cooperative_tensor<decltype(tp0), decltype(tv), float>();
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] = 0.0f;
    const uint sgx = sg & 1, sgy = sg >> 1;
    const uint r0 = ((lane >> 1) & 3) | ((lane >> 4) << 2) | (sgy << 4);
    const uint c_base = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1)) + 16 * sgx;
    const bool writer = (lane & 9) == 0;
    const uint last = a.q_pos0 + min(q0 + R, a.n_q) - 1;
    const uint n_keys = last >= a.block ? a.n_k : min(a.n_k, last + 1);
    const float scale = a.scale * 1.4426950408889634f;
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint buf = 0;
    load_k(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j0 = 0; j0 < n_keys; j0 += K) {
        for (uint16_t i = 0; i < CAP; ++i)
            sc[i] = 0.0f;
        qk.run(tq, tk, sc);
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
        // Everyone is past this pass's Q K^T: the next pass's keys and this pass's values go in.
        if (j0 + K < n_keys)
            load_k(j0 + K);
        load_v(j0);
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
        if (buf == 0)
            pv.run(tp0, tv, acc);
        else
            pv.run(tp1, tv, acc);
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
    for (uint16_t i = 0; i < ACAP; ++i)
        acc[i] *= inv[((i >> 2) & 1) | ((i / (8 * AQ)) << 1)];
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(o + uint64_t(q0) * a.width + h * D,
                                                                 dextents<int32_t, 2>(int(a.width), int(a.n_q - q0)));
    auto ot = to.slice(0, 0);
    acc.store(ot);
}
template [[host_name("xp_flash_attn_coop_tg_128_32")]] kernel void flash_attn_coop_tg<128, 32>(
    device const half*, device const half*, device const half*, device float*, constant AttnArgs&, uint2, ushort, ushort,
    ushort);

// The rate test for other element types: bfloat and half against int8 with an int32 accumulator (the hardware's int8
// rate against BF16's is the question behind int8 GEMMs).
template <typename T, typename ACC, int M, int N, int Dp, int NSG, bool COOP>
kernel void mm_rate_t(device const T* q [[buffer(0)]], device const T* k [[buffer(1)]], device ACC* o [[buffer(2)]],
                      constant RateArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]])
{
    const uint64_t qoff = COOP ? uint64_t(tg.x) * M * Dp : (uint64_t(tg.x) * NSG + sg) * M * Dp;
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tq(const_cast<device T*>(q) + qoff,
                                                             dextents<int32_t, 2>(Dp, M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tk(const_cast<device T*>(k),
                                                             dextents<int32_t, 2>(Dp, int(a.k_pad)));
    constexpr auto desc = matmul2d_descriptor(M, N, Dp, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<COOP ? NSG : 1>> op;
    auto kt0 = tk.slice(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tq), decltype(kt0), ACC>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        acc[i] = ACC(0);
    for (uint it = 0; it < a.iters; ++it) {
        auto kt = tk.slice(0, int((it * N) % a.k_pad));
        op.run(tq, kt, acc);
    }
    tensor<device ACC, dextents<int32_t, 2>, tensor_inline> to(o + qoff / Dp * N, dextents<int32_t, 2>(N, M));
    acc.store(to.slice(0, 0));
}
#define MM_RATE_T(T, ACC, m, n, d, nsg, coop, name)                                                                  \
    template [[host_name(name)]] kernel void mm_rate_t<T, ACC, m, n, d, nsg, coop>(                                  \
        device const T*, device const T*, device ACC*, constant RateArgs&, uint2, ushort);
MM_RATE_T(bfloat, float, 64, 128, 128, 4, true, "mm_rate_bf16_64x128x128_coop4")
MM_RATE_T(int8_t, int32_t, 64, 128, 128, 4, true, "mm_rate_i8_64x128x128_coop4")
MM_RATE_T(bfloat, float, 128, 128, 128, 8, true, "mm_rate_bf16_128x128x128_coop8")
MM_RATE_T(half, float, 128, 128, 128, 8, true, "mm_rate_f16_128x128x128_coop8")
MM_RATE_T(int8_t, int32_t, 128, 128, 128, 8, true, "mm_rate_i8_128x128x128_coop8")
MM_RATE_T(int8_t, int32_t, 64, 64, 128, 4, true, "mm_rate_i8_64x64x128_coop4")
