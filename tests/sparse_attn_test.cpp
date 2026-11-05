// The block-sparse attention against a CPU reference: cube_select's pooled scores and top-k, then attention_sparse
// over 6 cubes (one with padding rows) and 100 dense rows, each cube's queries over its selected cubes and the dense
// rows, the dense rows' queries over everything; rows mapped back through an identity map with the padding dropped.
//   sparse_attn_test
#include "models/runtime.hpp"
#include "mtl/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <print>
#include <random>
#include <vector>

namespace {

class Harness final : public mako::models::TransformerRuntime {
public:
    Harness(mako::mtl::Context& ctx) : TransformerRuntime(ctx, mako::gpu::Telemetry::off) {}
    const char* name() const override { return "sparse attention test"; }
    using TransformerRuntime::attention_sparse;
    using TransformerRuntime::cube_select;
    using TransformerRuntime::ctx_;
    using TransformerRuntime::cur_cb_;
    using TransformerRuntime::encoder;
    using TransformerRuntime::finish;
    using TransformerRuntime::scratch;
};

int failures = 0;
void check(bool ok, const std::string& what)
{
    std::println("  {}  {}", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    mako::mtl::Context ctx;
    Harness h(ctx);
    constexpr uint32_t D = 128, heads = 2, cubes = 6, dense = 100, topk = 3;
    const uint32_t cnt[cubes] = {64, 64, 64, 64, 64, 20};
    const uint32_t n = cubes * 64 + dense, q_pad = (n + 63) / 64 * 64, k_pad = (n + 127) / 128 * 128;
    const uint32_t dense0 = cubes * 64, dense1 = n;
    std::vector<uint8_t> valid(k_pad, 0);
    std::vector<int32_t> map(q_pad, -1);
    for (uint32_t c = 0; c < cubes; ++c)
        for (uint32_t r = 0; r < cnt[c]; ++r)
            valid[c * 64 + r] = 1;
    for (uint32_t r = dense0; r < dense1; ++r)
        valid[r] = 1;
    for (uint32_t r = 0; r < n; ++r)
        if (valid[r])
            map[r] = int32_t(r);
    // Random half tiles [heads][k_pad][D]; invalid rows zero, as the mapped heads kernel writes them.
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> qf(uint64_t(heads) * k_pad * D), kf(qf.size()), vf(qf.size());
    auto qb = h.scratch("t_q", qf.size() * 2), kb = h.scratch("t_k", qf.size() * 2), vb = h.scratch("t_v", qf.size() * 2);
    for (auto [f, b] : {std::pair{&qf, qb}, std::pair{&kf, kb}, std::pair{&vf, vb}}) {
        auto* hp = static_cast<__fp16*>(b->contents());
        for (uint32_t hh = 0; hh < heads; ++hh)
            for (uint32_t r = 0; r < k_pad; ++r)
                for (uint32_t d = 0; d < D; ++d) {
                    const uint64_t i = (uint64_t(hh) * k_pad + r) * D + d;
                    const __fp16 x = __fp16(valid[r] ? nd(rng) : 0.0f);
                    hp[i] = x;
                    (*f)[i] = float(x);
                }
    }
    auto cntb = h.scratch("t_cnt", cubes * 4);
    std::memcpy(cntb->contents(), cnt, sizeof(cnt));
    auto validb = h.scratch("t_valid", k_pad);
    std::memcpy(validb->contents(), valid.data(), valid.size());
    auto mapb = h.scratch("t_map", q_pad * 4);
    std::memcpy(mapb->contents(), map.data(), map.size() * 4);
    auto ob = h.scratch("t_o", uint64_t(q_pad) * heads * D * 4);
    auto* of = static_cast<float*>(ob->contents());
    std::fill_n(of, uint64_t(q_pad) * heads * D, 12345.0f);
    MTL::Buffer *sel = nullptr, *nsel = nullptr;
    {
        MTL::CommandBuffer* cb = h.ctx_.queue()->commandBuffer();
        h.cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = h.encoder(cb, 0, -1);
        h.cube_select(enc, qb, kb, heads, k_pad, cubes, cntb, topk, sel, nsel);
        h.attention_sparse(enc, qb, kb, vb, ob, n, q_pad, n, k_pad, heads, heads, heads * D, cubes, topk, dense0, dense1,
                           sel, nsel, validb, mapb);
        enc->endEncoding();
        h.finish(cb);
    }
    // The selection against the CPU's.
    const auto* gsel = static_cast<const uint16_t*>(sel->contents());
    const auto* gnsel = static_cast<const uint16_t*>(nsel->contents());
    std::vector<std::vector<uint32_t>> chosen(heads * cubes);
    uint32_t sel_bad = 0;
    for (uint32_t hh = 0; hh < heads; ++hh) {
        std::vector<std::vector<double>> qp(cubes, std::vector<double>(D, 0.0)), kp = qp;
        for (uint32_t c = 0; c < cubes; ++c)
            for (uint32_t r = 0; r < cnt[c]; ++r)
                for (uint32_t d = 0; d < D; ++d) {
                    qp[c][d] += qf[(uint64_t(hh) * k_pad + c * 64 + r) * D + d] / cnt[c];
                    kp[c][d] += kf[(uint64_t(hh) * k_pad + c * 64 + r) * D + d] / cnt[c];
                }
        for (uint32_t qc = 0; qc < cubes; ++qc) {
            std::vector<std::pair<double, uint32_t>> sc;
            for (uint32_t kc = 0; kc < cubes; ++kc) {
                double s = 0;
                for (uint32_t d = 0; d < D; ++d)
                    s += qp[qc][d] * kp[kc][d];
                sc.emplace_back(-s, kc);
            }
            std::ranges::sort(sc);
            std::vector<uint32_t> want, got;
            for (uint32_t i = 0; i < topk; ++i) {
                want.push_back(sc[i].second);
                got.push_back(gsel[(uint64_t(hh) * cubes + qc) * topk + i]);
            }
            chosen[hh * cubes + qc] = got;
            sel_bad += want != got || gnsel[hh * cubes + qc] != topk;
        }
    }
    check(sel_bad == 0, std::format("cube_select: {} of {} selections differ from the CPU's", sel_bad, heads * cubes));
    // The attention against the CPU's, over the GPU's selection.
    double worst = 0;
    uint64_t untouched_bad = 0;
    const float scale = 1.0f / std::sqrt(float(D));
    std::vector<double> ref(D);
    for (uint32_t hh = 0; hh < heads; ++hh)
        for (uint32_t r = 0; r < q_pad; ++r) {
            const float* out = of + uint64_t(r) * heads * D + hh * D;
            if (r >= n || !valid[r]) {
                for (uint32_t d = 0; d < D; ++d)
                    untouched_bad += out[d] != 12345.0f;
                continue;
            }
            std::vector<uint32_t> keys;
            if (r < dense0) {
                for (uint32_t c : chosen[hh * cubes + r / 64])
                    for (uint32_t j = c * 64; j < c * 64 + 64; ++j)
                        if (valid[j])
                            keys.push_back(j);
                for (uint32_t j = dense0; j < dense1; ++j)
                    keys.push_back(j);
            } else {
                for (uint32_t j = 0; j < n; ++j)
                    if (valid[j])
                        keys.push_back(j);
            }
            std::vector<double> w(keys.size());
            double mx = -1e30;
            for (size_t i = 0; i < keys.size(); ++i) {
                double s = 0;
                for (uint32_t d = 0; d < D; ++d)
                    s += double(qf[(uint64_t(hh) * k_pad + r) * D + d]) * kf[(uint64_t(hh) * k_pad + keys[i]) * D + d];
                w[i] = s * scale;
                mx = std::max(mx, w[i]);
            }
            double sum = 0;
            for (double& x : w) {
                x = std::exp(x - mx);
                sum += x;
            }
            std::fill(ref.begin(), ref.end(), 0.0);
            for (size_t i = 0; i < keys.size(); ++i)
                for (uint32_t d = 0; d < D; ++d)
                    ref[d] += w[i] / sum * vf[(uint64_t(hh) * k_pad + keys[i]) * D + d];
            double e = 0, nrm = 0;
            for (uint32_t d = 0; d < D; ++d) {
                e += (out[d] - ref[d]) * (out[d] - ref[d]);
                nrm += ref[d] * ref[d];
            }
            worst = std::max(worst, std::sqrt(e / std::max(nrm, 1e-30)));
        }
    check(worst < 5e-3, std::format("attention_sparse: worst row rel error {:.2e}", worst));
    check(untouched_bad == 0, std::format("padding and out-of-range rows left alone ({} values written)", untouched_bad));
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
