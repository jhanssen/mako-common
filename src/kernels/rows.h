// Pieces shared by row kernels: threadgroups of TG threads, their sums, the sigmoid and the RMS norm's arguments.
#pragma once
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

// Sum of `v` over a threadgroup of TG threads; every thread gets the total.
[[maybe_unused]] static float tg_sum(float v, threadgroup float* scratch, uint tid)
{
    v = simd_sum(v);
    if (tid % 32 == 0)
        scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32;
    float total = lane < TG / 32 ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

[[maybe_unused]] static float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

struct NormArgs {
    uint n;       // row length
    uint wn;      // weight length (the weight repeats every wn elements); 0: no weight
    float eps;
    uint ystride; // rms_norm: y row stride in elements (0: n)
};
