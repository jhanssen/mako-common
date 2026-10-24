// GEMV kernels for shapes the per-format kernels handle poorly; fp32 activations.
#include <metal_stdlib>
using namespace metal;

struct ShortArgs {
    uint k, n, row_bytes;
};

// Q8_0 block: a half scale and 32 int8 quants.
static float q8_dot(device const uchar* blk, threadgroup const float* x)
{
    const float d = float(*reinterpret_cast<device const half*>(blk));
    device const char* q = reinterpret_cast<device const char*>(blk + 2);
    float acc = 0.0f;
    for (uint i = 0; i < 32; ++i)
        acc += float(q[i]) * x[i];
    return d * acc;
}

// Short rows (k <= 512), which the per-format kernels spread over 32 lanes per row and leave mostly idle: four
// lanes per row, x staged in threadgroup memory. Threadgroups of 256 threads (64 rows). F32 or Q8_0 weights.
template <bool Q8>
kernel void gemv_short(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]], constant ShortArgs& a [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]], uint tg [[threadgroup_position_in_grid]])
{
    threadgroup float xs[512];
    for (uint i = tid; i < a.k; i += 256)
        xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = tg * 64 + tid / 4, lane = tid % 4;
    float acc = 0.0f;
    if (row < a.n) {
        device const uchar* r = w + uint64_t(row) * a.row_bytes;
        if (Q8) {
            for (uint b = lane; b < a.k / 32; b += 4)
                acc += q8_dot(r + b * 34, xs + b * 32);
        } else {
            device const float4* r4 = reinterpret_cast<device const float4*>(r);
            threadgroup const float4* x4 = reinterpret_cast<threadgroup const float4*>(xs);
            for (uint i = lane; i < a.k / 4; i += 4)
                acc += dot(r4[i], x4[i]);
        }
    }
    acc = quad_sum(acc);
    if (row < a.n && lane == 0)
        y[row] = acc;
}

template [[host_name("gemv_short_f32")]] kernel void gemv_short<false>(device const uchar*, device const float*,
                                                                       device float*, constant ShortArgs&, uint, uint);
template [[host_name("gemv_short_q8_0")]] kernel void gemv_short<true>(device const uchar*, device const float*,
                                                                       device float*, constant ShortArgs&, uint, uint);
