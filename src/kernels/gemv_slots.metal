// Indexed GEMV over expert slots kept in two buffers: slots below `split` in src0s, the rest in src0t (the arena's
// lendable slots). Otherwise as the per-format indexed GEMV it wraps.
#include "mul_mv.metal"

template <mul_mv_disp_fn_t disp_fn>
kernel void mako_mul_mv_slots(constant ggml_metal_kargs_mul_mv_id& args, device const char* src0s,
                              device const char* src1, device char* dst, device const char* ids,
                              device const char* src0t, constant uint& split,
                              threadgroup char* shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],
                              ushort tiitg [[thread_index_in_threadgroup]], ushort tiisg [[thread_index_in_simdgroup]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int iid1 = tgpig.z / args.nei0;
    const int idx = tgpig.z % args.nei0;
    tgpig.z = 0;
    const int32_t i02 = ((device const int32_t*)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char* src0_cur = uint(i02) < split ? src0s + i02 * args.nb02 : src0t + (i02 - int(split)) * args.nb02;
    device const char* src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char* dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ggml_metal_kargs_mul_mv args0 = {
        args.ne00, args.ne01, 1, args.nb00, args.nb01, args.nb02, args.nb02, args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12, args.ne0, 1, args.nr0, 1, 1,
    };
    disp_fn(args0, src0_cur, src1_cur, dst_cur, shmem, tgpig, tiitg, tiisg, sgitg);
}

// Q6_K rows for NT tokens at once (x rows nb11 apart, y rows ne0 apart): the per-format kernel's layout and per-token
// arithmetic (so the same results), but each weight block is decoded once for all the tokens instead of being
// read again per token. Grid (row groups, 1, matrices).
template <int NT>
kernel void mako_mul_mv_q6_K_tokens(constant ggml_metal_kargs_mul_mv& args, device const char* src0,
                                    device const char* src1, device char* dst,
                                    uint3 tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short nr0 = N_R0_Q6_K;
    const short NSG = FC_mul_mv_nsg;
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = args.ne00 / QK_K;
    const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
    const uint im = tgpig.z;
    const uint i12 = im % FC_mul_mv_ne12, i13 = im / FC_mul_mv_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03;
    const uint64_t offset1 = i12 * args.nb12 + i13 * args.nb13;
    device const block_q6_K* x = (device const block_q6_K*)(src0 + offset0);

    float sumf[NT][nr0] = {};
    float yl[NT][16];
    const short tid = tiisg / 2, ix = tiisg % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t* q1 = x[i].ql + q_offset_l;
        device const uint8_t* q2 = q1 + 32;
        device const uint8_t* qh = x[i].qh + q_offset_h;
        device const int8_t* sc = x[i].scales + is;
        device const half* dh = &x[i].d;
        for (short t = 0; t < NT; ++t) {
            device const float* y = (device const float*)(src1 + offset1 + t * args.nb11) + i * QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[t][4 * l + 0] = y[l + 0];
                yl[t][4 * l + 1] = y[l + 32];
                yl[t][4 * l + 2] = y[l + 64];
                yl[t][4 * l + 3] = y[l + 96];
            }
        }
        for (short row = 0; row < nr0; ++row) {
            float q[4][4];
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                q[l][0] = (int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32;
                q[l][1] = (int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32;
                q[l][2] = (int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32;
                q[l][3] = (int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32;
            }
            const float d = dh[0];
            for (short t = 0; t < NT; ++t) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};
                FOR_UNROLL (short l = 0; l < 4; ++l)
                    for (short c = 0; c < 4; ++c)
                        sums[c] += yl[t][4 * l + c] * q[l][c];
                sumf[t][row] += d * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            }
            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }
    for (short t = 0; t < NT; ++t) {
        device float* dst_f32 = (device float*)dst + uint64_t(im) * args.ne0 * args.ne1 + uint64_t(t) * args.ne0;
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[t][row]);
            if (tiisg == 0)
                dst_f32[first_row + row] = sum_all;
        }
    }
}

template [[host_name("mako_mul_mv_q6_K_t2")]] kernel void mako_mul_mv_q6_K_tokens<2>(constant ggml_metal_kargs_mul_mv&,
    device const char*, device const char*, device char*, uint3, ushort, ushort);
template [[host_name("mako_mul_mv_q6_K_t3")]] kernel void mako_mul_mv_q6_K_tokens<3>(constant ggml_metal_kargs_mul_mv&,
    device const char*, device const char*, device char*, uint3, ushort, ushort);
template [[host_name("mako_mul_mv_q6_K_t4")]] kernel void mako_mul_mv_q6_K_tokens<4>(constant ggml_metal_kargs_mul_mv&,
    device const char*, device const char*, device char*, uint3, ushort, ushort);

typedef decltype(mako_mul_mv_slots<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>) mako_mul_mv_slots_t;
typedef decltype(mako_mul_mv_slots<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>) mako_mul_mv_slots_4_t;

#define SLOTS(name, ...) template [[host_name("mako_mul_mv_slots_" name)]] kernel mako_mul_mv_slots_t mako_mul_mv_slots<mmv_fn<__VA_ARGS__>>;
template [[host_name("mako_mul_mv_slots_f32_f32")]] kernel mako_mul_mv_slots_t mako_mul_mv_slots<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>;
template [[host_name("mako_mul_mv_slots_f32_f32_4")]] kernel mako_mul_mv_slots_4_t mako_mul_mv_slots<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>;
SLOTS("q8_0_f32", kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>)
SLOTS("q4_0_f32", mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>)
SLOTS("mxfp4_f32", kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>)
SLOTS("q2_K_f32", kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>)
SLOTS("q3_K_f32", kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K>)
SLOTS("q4_K_f32", kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>)
SLOTS("q5_K_f32", kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K>)
SLOTS("q6_K_f32", kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K>)
SLOTS("iq2_xxs_f32", kernel_mul_mv_iq2_xxs_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq2_xs_f32", kernel_mul_mv_iq2_xs_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq2_s_f32", kernel_mul_mv_iq2_s_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq3_xxs_f32", kernel_mul_mv_iq3_xxs_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq3_s_f32", kernel_mul_mv_iq3_s_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq4_xs_f32", kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS>)
SLOTS("iq1_s_f32", kernel_mul_mv_iq1_s_f32_disp<ggml_metal_kargs_mul_mv>)
SLOTS("iq1_m_f32", kernel_mul_mv_iq1_m_f32_disp<ggml_metal_kargs_mul_mv>)

// IQ2_XXS and IQ3_XXS rows with the per-format kernels' layout and each lane's summation order, but the grid entries
// converted to half once per threadgroup (eight weights are two loads instead of eight byte loads and conversions) and
// the signs applied by negation. Two simdgroups of nr0 rows each. IQ2_XXS matches the per-format kernel bit for bit;
// IQ3_XXS's arithmetic is fused differently by the compiler (~1e-7 relative).
template <int nr0, typename args_t>
void mako_iq2_xxs_impl(args_t args, device const char* src0, device const char* src1, device char* dst,
                       threadgroup char* shmem, uint3 tgpig, ushort tiisg, ushort sgitg)
{
    const short NSG = FC_mul_mv_nsg;
    const int nb32 = args.ne00 / 32;
    const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
    const uint im = tgpig.z, i12 = im % FC_mul_mv_ne12, i13 = im / FC_mul_mv_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03;
    const uint64_t offset1 = tgpig.y * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_iq2_xxs* x = (device const block_iq2_xxs*)(src0 + offset0);
    device const float4* y4 = (device const float4*)(src1 + offset1) + 8 * tiisg;

    threadgroup half4* sgrid = (threadgroup half4*)shmem; // 256 entries of 8
    threadgroup uint8_t* ssigns = (threadgroup uint8_t*)(sgrid + 512);
    for (short i = 32 * sgitg + tiisg; i < 256; i += 32 * NSG) {
        const uint64_t g = iq2xxs_grid[i];
        sgrid[2 * i] = half4(half(g & 255), half((g >> 8) & 255), half((g >> 16) & 255), half((g >> 24) & 255));
        sgrid[2 * i + 1] = half4(half((g >> 32) & 255), half((g >> 40) & 255), half((g >> 48) & 255), half(g >> 56));
    }
    for (short i = 32 * sgitg + tiisg; i < 128; i += 32 * NSG)
        ssigns[i] = ksigns_iq2xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float yl[32];
    float sumf[nr0] = {0.f};
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 8; ++i) {
            const float4 v = y4[i];
            yl[4 * i] = v[0];
            yl[4 * i + 1] = v[1];
            yl[4 * i + 2] = v[2];
            yl[4 * i + 3] = v[3];
        }
        device const block_iq2_xxs* xr = x + ib32 / 8;
        device const uint16_t* q2 = xr->qs + 4 * (ib32 % 8);
        device const half* dh = &xr->d;
        for (short row = 0; row < nr0; ++row) {
            const float db = dh[0];
            device const uint8_t* aux8 = (device const uint8_t*)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));
            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const half4 g0 = sgrid[2 * aux8[l]], g1 = sgrid[2 * aux8[l] + 1];
                const uint signs = ssigns[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 4; ++j) {
                    const float p = yl[8 * l + j] * float(g0[j]);
                    sum += (signs >> j) & 1 ? -p : p;
                }
                for (short j = 0; j < 4; ++j) {
                    const float p = yl[8 * l + 4 + j] * float(g1[j]);
                    sum += (signs >> (4 + j)) & 1 ? -p : p;
                }
            }
            sumf[row] += d * sum;
            dh += args.nb01 / 2;
            q2 += args.nb01 / 2;
        }
        y4 += 8 * 32;
    }
    device float* dst_f32 = (device float*)dst + uint64_t(im) * args.ne0 * args.ne1 + uint64_t(tgpig.y) * args.ne0;
    for (short row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0)
            dst_f32[first_row + row] = s * 0.25f;
    }
}

template <int nr0, typename args_t>
void mako_iq3_xxs_impl(args_t args, device const char* src0, device const char* src1, device char* dst,
                       threadgroup char* shmem, uint3 tgpig, ushort tiisg, ushort sgitg)
{
    const short NSG = FC_mul_mv_nsg;
    const int nb32 = args.ne00 / 32;
    const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
    const uint im = tgpig.z, i12 = im % FC_mul_mv_ne12, i13 = im / FC_mul_mv_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03;
    const uint64_t offset1 = tgpig.y * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_iq3_xxs* x = (device const block_iq3_xxs*)(src0 + offset0);
    device const float4* y4 = (device const float4*)(src1 + offset1) + 8 * tiisg;
    threadgroup half4* sgrid = (threadgroup half4*)shmem;
    threadgroup uint8_t* ssigns = (threadgroup uint8_t*)(sgrid + 256);
    for (short i = 32 * sgitg + tiisg; i < 256; i += 32 * NSG) {
        const uint32_t g = iq3xxs_grid[i];
        sgrid[i] = half4(half(g & 255), half((g >> 8) & 255), half((g >> 16) & 255), half(g >> 24));
    }
    for (short i = 32 * sgitg + tiisg; i < 128; i += 32 * NSG)
        ssigns[i] = ksigns_iq2xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float yl[32];
    float sumf[nr0] = {0.f};
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 8; ++i) {
            const float4 v = y4[i];
            yl[4 * i] = v[0];
            yl[4 * i + 1] = v[1];
            yl[4 * i + 2] = v[2];
            yl[4 * i + 3] = v[3];
        }
        device const block_iq3_xxs* xr = x + ib32 / 8;
        const int ib = ib32 % 8;
        device const uint8_t* q3 = xr->qs + 8 * ib;
        device const uint16_t* gas = (device const uint16_t*)(xr->qs + QK_K / 4) + 2 * ib;
        device const half* dh = &xr->d;
        for (short row = 0; row < nr0; ++row) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));
            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const half4 g1 = sgrid[q3[2 * l]], g2 = sgrid[q3[2 * l + 1]];
                const uint signs = ssigns[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 4; ++j) {
                    const float p1 = yl[8 * l + j] * float(g1[j]), p2 = yl[8 * l + 4 + j] * float(g2[j]);
                    sum[0] += (signs >> j) & 1 ? -p1 : p1;
                    sum[1] += (signs >> (4 + j)) & 1 ? -p2 : p2;
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
            dh += args.nb01 / 2;
            q3 += args.nb01;
            gas += args.nb01 / 2;
        }
        y4 += 8 * 32;
    }
    device float* dst_f32 = (device float*)dst + uint64_t(im) * args.ne0 * args.ne1 + uint64_t(tgpig.y) * args.ne0;
    for (short row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0)
            dst_f32[first_row + row] = s * 0.5f;
    }
}

// IQ1_S and IQ1_M rows as the per-format kernels compute them (per 8 weights a grid entry of 2-bit values in nibbles,
// an odd scale and a shift times the sum of those 8 activations), but each grid entry one 32-bit load and its eight
// values two dot products (the per-format kernels: byte loads and scalar products; decode is arithmetic-bound here).
template <int nr0, typename args_t>
void mako_iq1_m_impl(args_t args, device const char* src0, device const char* src1, device char* dst,
                     threadgroup char* shmem, uint3 tgpig, ushort tiisg, ushort sgitg)
{
    const short NSG = FC_mul_mv_nsg;
    const int nb32 = args.ne00 / 32;
    const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
    const uint im = tgpig.z, i12 = im % FC_mul_mv_ne12, i13 = im / FC_mul_mv_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03;
    const uint64_t offset1 = tgpig.y * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_iq1_m* x = (device const block_iq1_m*)(src0 + offset0);
    device const float4* y4 = (device const float4*)(src1 + offset1) + 8 * tiisg;
    constant uint32_t* sgrid = iq1s_grid_gpu;

    float4 yv[8];
    float sumf[nr0] = {0.f};
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        float4 sumy = 0.f;
        for (short i = 0; i < 8; ++i) {
            yv[i] = y4[i];
            sumy[i / 2] += yv[i][0] + yv[i][1] + yv[i][2] + yv[i][3];
        }
        device const block_iq1_m* xr = x + ib32 / 8;
        const int ib = ib32 % 8;
        device const uint8_t* qs = xr->qs + 4 * ib;
        device const uint8_t* qh = xr->qh + 2 * ib;
        device const uint16_t* sc = (device const uint16_t*)xr->scales;
        for (short row = 0; row < nr0; ++row) {
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const uint32_t g[4] = {sgrid[qs[0] | ((qh[0] << 8) & 0x700)], sgrid[qs[1] | ((qh[0] << 4) & 0x700)],
                                   sgrid[qs[2] | ((qh[1] << 8) & 0x700)], sgrid[qs[3] | ((qh[1] << 4) & 0x700)]};
            // nibbles of entry l: weights 8l..8l+3 low, 8l+4..8l+7 high
            float2 sum = 0.f;
            for (short l = 0; l < 4; ++l)
                sum[l / 2] += dot(yv[2 * l], float4(as_type<uchar4>(g[l] & 0x0f0f0f0fu))) +
                              dot(yv[2 * l + 1], float4(as_type<uchar4>((g[l] >> 4) & 0x0f0f0f0fu)));
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) +
                                 sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) +
                                 sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            sumf[row] += float(scale.f16) * ((sum[0] + delta1) * float(2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) + 1) +
                                             (sum[1] + delta2) * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1));
            sc += args.nb01 / 2;
            qs += args.nb01;
            qh += args.nb01;
        }
        y4 += 8 * 32;
    }
    device float* dst_f32 = (device float*)dst + uint64_t(im) * args.ne0 * args.ne1 + uint64_t(tgpig.y) * args.ne0;
    for (short row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0)
            dst_f32[first_row + row] = s;
    }
}

template <int nr0, typename args_t>
void mako_iq1_s_impl(args_t args, device const char* src0, device const char* src1, device char* dst,
                     threadgroup char* shmem, uint3 tgpig, ushort tiisg, ushort sgitg)
{
    const short NSG = FC_mul_mv_nsg;
    const int nb32 = args.ne00 / 32;
    const int first_row = (int(tgpig.x) * NSG + sgitg) * nr0;
    const uint im = tgpig.z, i12 = im % FC_mul_mv_ne12, i13 = im / FC_mul_mv_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03;
    const uint64_t offset1 = tgpig.y * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_iq1_s* x = (device const block_iq1_s*)(src0 + offset0);
    device const float4* y4 = (device const float4*)(src1 + offset1) + 8 * tiisg;
    constant uint32_t* sgrid = iq1s_grid_gpu;

    float4 yv[8];
    float sumf[nr0] = {0.f};
    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        float sumy = 0.f;
        for (short i = 0; i < 8; ++i) {
            yv[i] = y4[i];
            sumy += yv[i][0] + yv[i][1] + yv[i][2] + yv[i][3];
        }
        device const block_iq1_s* xr = x + ib32 / 8;
        const int ib = ib32 % 8;
        device const uint8_t* qs = xr->qs + 4 * ib;
        device const uint16_t* qh = xr->qh + ib;
        device const half* dh = &xr->d;
        for (short row = 0; row < nr0; ++row) {
            const uint h = qh[0];
            const uint32_t g[4] = {sgrid[qs[0] | ((h << 8) & 0x700)], sgrid[qs[1] | ((h << 5) & 0x700)],
                                   sgrid[qs[2] | ((h << 2) & 0x700)], sgrid[qs[3] | ((h >> 1) & 0x700)]};
            float sum = 0.f;
            for (short l = 0; l < 4; ++l)
                sum += dot(yv[2 * l], float4(as_type<uchar4>(g[l] & 0x0f0f0f0fu))) +
                       dot(yv[2 * l + 1], float4(as_type<uchar4>((g[l] >> 4) & 0x0f0f0f0fu)));
            sumf[row] += float(dh[0]) * float(2 * ((h >> 12) & 7) + 1) *
                         (sum + sumy * (h & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA));
            dh += args.nb01 / 2;
            qs += args.nb01;
            qh += args.nb01 / 2;
        }
        y4 += 8 * 32;
    }
    device float* dst_f32 = (device float*)dst + uint64_t(im) * args.ne0 * args.ne1 + uint64_t(tgpig.y) * args.ne0;
    for (short row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0)
            dst_f32[first_row + row] = s;
    }
}

#define MAKO_IQ(fmt, nr)                                                                                               \
    [[host_name("mako_mul_mv_" #fmt "_f32_r" #nr)]] kernel void mako_mul_mv_##fmt##_r##nr(                          \
        constant ggml_metal_kargs_mul_mv& args, device const char* src0, device const char* src1, device char* dst,   \
        threadgroup char* shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],                     \
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]])                  \
    {                                                                                                                  \
        mako_##fmt##_impl<nr, constant ggml_metal_kargs_mul_mv&>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);   \
    }                                                                                                                  \
    SLOTS("mako_" #fmt "_f32_r" #nr, mako_##fmt##_impl<nr>)
MAKO_IQ(iq2_xxs, 4)
MAKO_IQ(iq3_xxs, 4)
MAKO_IQ(iq1_m, 4)
MAKO_IQ(iq1_s, 4)

#include "moe_route.h"

// The F32 router GEMV (a few tokens, rows of n_expert <= 512) that also routes: the last threadgroup to finish a
// token's rows (a device-scope fence, then a counter per token, zero between dispatches) selects its experts as
// moe_route does. Buffers 0-3 as the GEMV, 4-13 as moe_route's 1-10, 14 the counters.
kernel void mako_mul_mv_f32_f32_4_route(constant ggml_metal_kargs_mul_mv& args [[buffer(0)]],
                                        device const char* src0 [[buffer(1)]], device const char* src1 [[buffer(2)]],
                                        device char* dst [[buffer(3)]], device const float* bias [[buffer(4)]],
                                        device int* sel [[buffer(5)]], device float* w [[buffer(6)]],
                                        device float* biased [[buffer(7)]], constant RouteArgs& ra [[buffer(8)]],
                                        device const int* slots [[buffer(9)]], device int* hit_ids [[buffer(10)]],
                                        device int* miss_rank [[buffer(11)]], device uint* indirect [[buffer(12)]],
                                        device atomic_uint* counts [[buffer(13)]],
                                        device atomic_uint* counters [[buffer(14)]],
                                        threadgroup char* shmem [[threadgroup(0)]],
                                        uint3 tgpig [[threadgroup_position_in_grid]],
                                        ushort tiitg [[thread_index_in_threadgroup]],
                                        ushort tiisg [[thread_index_in_simdgroup]],
                                        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2, constant ggml_metal_kargs_mul_mv&>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
    threadgroup uint last[1];
    const uint t = tgpig.y, tgs = (uint(args.ne01) + 1) / 2; // threadgroups per token: 2 rows each
    threadgroup_barrier(mem_flags::mem_device);
    if (tiitg == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last[0] = atomic_fetch_add_explicit(counters + t, 1u, memory_order_relaxed) + 1 == tgs ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last[0])
        return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (sgitg == 0) {
        device const float* logits = reinterpret_cast<device const float*>(dst) + uint64_t(t) * uint(args.ne0);
        moe_route_token(logits, bias, sel + t * 16, w + t * 16, biased + uint64_t(t) * ra.n_expert, ra, slots,
                        hit_ids + t * 16, miss_rank + t * 16, indirect + t * 9, counts, tiisg);
    }
    if (tiitg == 0)
        atomic_store_explicit(counters + t, 0u, memory_order_relaxed);
}
