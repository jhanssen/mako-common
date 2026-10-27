// Router top-k for one token, shared by the moe_route kernel and the router GEMV that routes in its last threadgroup.
#pragma once

struct RouteArgs {
    uint n_expert, n_used, norm;
    float scale;
    uint lookup; // 1: look the selection up in slots (one token); 2: also count it in `counts`
    uint score;  // 0: sigmoid, 1: sqrt(softplus)
    float miss_penalty; // > 0: subtracted from the selection score (not the weight) of an expert slots lacks
};

static float route_score(float l, uint kind)
{
    if (kind == 0)
        return 1.0f / (1.0f + precise::exp(-l));
    return precise::sqrt(l > 20.0f ? l : precise::log(1.0f + precise::exp(l)));
}

// Top-k over sigmoid or sqrt(softplus) scores, the bias added for the selection only; ties keep the lower expert index.
// Weights are the unbiased scores of the selected experts, normalized when a.norm, times a.scale.
// Then each selected expert is looked up in `slots` (this layer's expert -> arena slot, -1 absent): a resident
// one gets its slot in hit_ids and miss_rank -1; a missing one slot 0 (its result is not used) and its index
// among the misses. The miss count goes to the z of the three indirect dispatches (gate, up, down).
// One simdgroup, lane l keeping experts l, l + 32, ... in registers (n_expert <= 512); each of the n_used rounds
// takes the largest biased score left (lowest index on ties). The pointers are at the token's rows: logits and
// biased of n_expert, sel, w, hit_ids and miss_rank of 16 (n_used <= 16 used), indirect of 9 (the last three only
// with a.lookup). Logits is a device float pointer, coherent(device) when other threadgroups of the dispatch wrote them.
template <typename Logits>
static void moe_route_token(Logits logits, device const float* bias, device int* sel, device float* w,
                            device float* biased, constant RouteArgs& a, device const int* slots,
                            device int* hit_ids, device int* miss_rank, device uint* indirect,
                            device atomic_uint* counts, uint lane)
{
    constexpr uint K = 16;
    float v[K];
    for (uint k = 0; k < K; ++k) {
        const uint e = lane + 32 * k;
        v[k] = -INFINITY;
        if (e < a.n_expert) {
            v[k] = route_score(logits[e], a.score) + bias[e];
            biased[e] = v[k];
            if (a.miss_penalty > 0.0f && slots[e] < 0)
                v[k] -= a.miss_penalty;
        }
    }
    uint mine = 0xffffffffu; // the expert chosen in round `lane`
    for (uint r = 0; r < a.n_used; ++r) {
        float bv = -INFINITY;
        uint bk = 0;
        for (uint k = 0; k < K; ++k)
            if (v[k] > bv) { // ascending e within the lane: strict > keeps the lowest index on ties
                bv = v[k];
                bk = k;
            }
        const float m = simd_max(bv);
        const uint e = simd_min(bv == m && bv > -INFINITY ? lane + 32 * bk : 0xffffffffu);
        if (e % 32 == lane)
            v[e / 32] = -INFINITY;
        if (lane == r)
            mine = e;
    }
    const bool used = lane < a.n_used;
    const float p = used ? route_score(logits[mine], a.score) : 0.0f;
    float sum = 0.0f;
    for (uint j = 0; j < a.n_used; ++j)
        sum += simd_shuffle(p, j);
    if (a.norm)
        sum = max(sum, 6.103515625e-5f);
    if (used) {
        sel[lane] = int(mine);
        w[lane] = (a.norm ? p / sum : p) * a.scale;
    }
    if (a.lookup) {
        const int s = used ? slots[mine] : 0;
        const uint miss = used && s < 0 ? 1u : 0u;
        const uint rank = simd_prefix_exclusive_sum(miss);
        if (used) {
            hit_ids[lane] = max(s, 0);
            miss_rank[lane] = miss ? int(rank) : -1;
        }
        if (a.lookup == 2 && used)
            atomic_fetch_add_explicit(&counts[mine], 1u, memory_order_relaxed);
        const uint misses = simd_sum(miss);
        if (lane < 3)
            indirect[lane * 3 + 2] = misses;
    }
}
