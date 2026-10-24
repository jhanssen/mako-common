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
