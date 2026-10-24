// Prefill GEMM: y[t][n] = sum_k W[n][k] * x[t][k] for many tokens. Each threadgroup computes 64 tokens x 64 rows:
// per 64-wide k step it dequantizes the weight tile and converts the activation tile to half in threadgroup
// memory, and matmul2d accumulates into a float cooperative tensor (the GPU's matrix units where it has them).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant int TM = 64, TN = 64, TK = 64;

struct GemmArgs {
    uint t, k, n, row_bytes;
};

enum WeightType { F32 = 0, Q8_0 = 1, Q6_K = 2 };

// Writes 32 weights of `row` starting at k0 (a multiple of 32) to out.
template <int Type>
static void dequant32(device const uchar* row, uint k0, threadgroup half* out)
{
    if (Type == F32) {
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

    tensor<device float, dextents<int32_t, 2>, tensor_inline> ty(y, dextents<int32_t, 2>(a.n, a.t));
    acc.store(ty.slice(n0, t0));
}

template [[host_name("gemm_f32")]] kernel void gemm<F32>(device const uchar*, device const float*, device float*,
                                                         constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q8_0")]] kernel void gemm<Q8_0>(device const uchar*, device const float*, device float*,
                                                           constant GemmArgs&, uint2, ushort);
template [[host_name("gemm_q6_k")]] kernel void gemm<Q6_K>(device const uchar*, device const float*, device float*,
                                                           constant GemmArgs&, uint2, ushort);
