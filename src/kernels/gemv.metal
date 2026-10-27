// GEMV kernels for shapes the per-format kernels handle poorly; fp32 activations.
#include <metal_stdlib>
using namespace metal;

struct ShortArgs {
    uint k, n, row_bytes;
    uint tokens; // 1..8: x rows of k, y rows of n
};

// Short rows (k <= 512), which the per-format kernels spread over 32 lanes per row and leave mostly idle: four
// lanes per row, x staged in threadgroup memory, each weight read once for all the tokens. Threadgroups of 256
// threads (64 rows). F32 or Q8_0 weights.
template <bool Q8>
kernel void gemv_short(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]], constant ShortArgs& a [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]], uint tg [[threadgroup_position_in_grid]])
{
    constexpr uint MT = 8;
    threadgroup float xs[MT][512];
    for (uint t = 0; t < a.tokens; ++t)
        for (uint i = tid; i < a.k; i += 256)
            xs[t][i] = x[uint64_t(t) * a.k + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = tg * 64 + tid / 4, lane = tid % 4;
    float acc[MT] = {};
    if (row < a.n) {
        device const uchar* r = w + uint64_t(row) * a.row_bytes;
        if (Q8) {
            for (uint b = lane; b < a.k / 32; b += 4) {
                device const uchar* blk = r + b * 34;
                const float d = float(*reinterpret_cast<device const half*>(blk));
                device const char* q = reinterpret_cast<device const char*>(blk + 2);
                for (uint t = 0; t < a.tokens; ++t) {
                    float s = 0.0f;
                    for (uint i = 0; i < 32; ++i)
                        s += float(q[i]) * xs[t][b * 32 + i];
                    acc[t] += d * s;
                }
            }
        } else {
            device const float4* r4 = reinterpret_cast<device const float4*>(r);
            for (uint i = lane; i < a.k / 4; i += 4) {
                const float4 wv = r4[i];
                for (uint t = 0; t < a.tokens; ++t)
                    acc[t] += dot(wv, reinterpret_cast<threadgroup const float4*>(xs[t])[i]);
            }
        }
    }
    for (uint t = 0; t < a.tokens; ++t) {
        const float v = quad_sum(acc[t]);
        if (row < a.n && lane == 0)
            y[uint64_t(t) * a.n + row] = v;
    }
}

template [[host_name("gemv_short_f32")]] kernel void gemv_short<false>(device const uchar*, device const float*,
                                                                       device float*, constant ShortArgs&, uint, uint);
template [[host_name("gemv_short_q8_0")]] kernel void gemv_short<true>(device const uchar*, device const float*,
                                                                       device float*, constant ShortArgs&, uint, uint);
