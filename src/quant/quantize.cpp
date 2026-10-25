#include "quant/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mako::quant {

namespace {

void put_f16(std::byte* p, float v)
{
    const _Float16 h = static_cast<_Float16>(v);
    std::memcpy(p, &h, 2);
}

float get_f16(const std::byte* p)
{
    _Float16 h;
    std::memcpy(&h, p, 2);
    return static_cast<float>(h);
}

int nearest(float v) { return int(std::lrintf(v)); }

// Scale and (negated) min of n values for levels 0..nmax, minimizing the weighted squared error: a first fit from
// the range, then a search over nearby scales with the least-squares scale and min for each rounding.
float fit_scale_min(int n, int nmax, const float* x, const float* w, uint8_t* L, float* neg_min)
{
    float lo = x[0], hi = x[0];
    for (int i = 1; i < n; ++i) {
        lo = std::min(lo, x[i]);
        hi = std::max(hi, x[i]);
    }
    lo = std::min(lo, 0.0f);
    if (hi <= lo) {
        std::fill_n(L, n, 0);
        *neg_min = -lo;
        return 0.0f;
    }
    float sum_w = 0, sum_x = 0;
    for (int i = 0; i < n; ++i) {
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    float iscale = nmax / (hi - lo), scale = 1.0f / iscale, best = 0;
    for (int i = 0; i < n; ++i) {
        const int l = std::clamp(nearest(iscale * (x[i] - lo)), 0, nmax);
        L[i] = uint8_t(l);
        const float d = scale * l + lo - x[i];
        best += w[i] * d * d;
    }
    uint8_t aux[32];
    for (int step = 0; step <= 20; ++step) {
        iscale = (-1.0f + 0.1f * step + nmax) / (hi - lo);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            const int l = std::clamp(nearest(iscale * (x[i] - lo)), 0, nmax);
            aux[i] = uint8_t(l);
            sum_l += w[i] * l;
            sum_l2 += w[i] * l * l;
            sum_xl += w[i] * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0)
            continue;
        float s = (sum_w * sum_xl - sum_x * sum_l) / D, m = (sum_l2 * sum_x - sum_l * sum_xl) / D;
        if (m > 0) {
            m = 0;
            s = sum_xl / sum_l2;
        }
        float err = 0;
        for (int i = 0; i < n; ++i) {
            const float d = s * aux[i] + m - x[i];
            err += w[i] * d * d;
        }
        if (err < best) {
            best = err;
            std::copy_n(aux, n, L);
            scale = s;
            lo = m;
        }
    }
    *neg_min = -lo;
    return scale;
}

// Symmetric scale of n values for levels -nmax..nmax-1 (stored as l + nmax), weighted by x^2, searched around the
// range's scale.
float fit_scale(int n, int nmax, const float* x, int8_t* L)
{
    float amax = 0, max = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(x[i]) > amax) {
            amax = std::fabs(x[i]);
            max = x[i];
        }
    if (amax < 1e-30f) {
        std::fill_n(L, n, int8_t(0));
        return 0.0f;
    }
    auto fit = [&](float iscale, int8_t* out, float& sumlx, float& suml2) {
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            const int l = std::clamp(nearest(iscale * x[i]), -nmax, nmax - 1);
            out[i] = int8_t(l);
            const float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
    };
    float sumlx, suml2;
    fit(-nmax / max, L, sumlx, suml2);
    float scale = suml2 > 0 ? sumlx / suml2 : 0.0f, best = scale * sumlx;
    int8_t aux[32];
    for (int is = -9; is <= 9; ++is) {
        if (is == 0)
            continue;
        fit(-(nmax + 0.1f * is) / max, aux, sumlx, suml2);
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            std::copy_n(aux, n, L);
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// The 6-bit scales and mins of the 8 sub-blocks of a Q4_K / Q5_K block, packed into 12 bytes.
void pack_scales(const uint8_t* ls, const uint8_t* lm, uint8_t* q)
{
    std::memset(q, 0, 12);
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            q[j] = ls[j];
            q[j + 4] = lm[j];
        } else {
            q[j + 4] = uint8_t((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
            q[j - 4] |= uint8_t((ls[j] >> 4) << 6);
            q[j] |= uint8_t((lm[j] >> 4) << 6);
        }
    }
}

void scale_min(int j, const uint8_t* q, int& sc, int& m)
{
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K / Q5_K block: d, dmin (f16), 12 bytes of scales, then (Q5_K) 32 bytes of high bits, then the low nibbles.
// Writes the 16-byte header and fills L with the levels 0..nmax.
void quantize_k(const float* x, int nmax, std::byte* blk, uint8_t* L)
{
    float scales[8], mins[8], w[32];
    for (int j = 0; j < 8; ++j) {
        const float* xs = x + 32 * j;
        float sum2 = 0;
        for (int i = 0; i < 32; ++i)
            sum2 += xs[i] * xs[i];
        const float av = std::sqrt(sum2 / 32);
        for (int i = 0; i < 32; ++i)
            w[i] = av + std::fabs(xs[i]);
        scales[j] = fit_scale_min(32, nmax, xs, w, L + 32 * j, &mins[j]);
    }
    const float max_scale = *std::max_element(scales, scales + 8), max_min = *std::max_element(mins, mins + 8);
    const float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f, inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    uint8_t ls[8], lm[8];
    for (int j = 0; j < 8; ++j) {
        ls[j] = uint8_t(std::min(63, nearest(inv_scale * scales[j])));
        lm[j] = uint8_t(std::min(63, nearest(inv_min * mins[j])));
    }
    put_f16(blk, max_scale / 63.0f);
    put_f16(blk + 2, max_min / 63.0f);
    auto* sq = reinterpret_cast<uint8_t*>(blk + 4);
    pack_scales(ls, lm, sq);
    const float d = get_f16(blk), dmin = get_f16(blk + 2);
    for (int j = 0; j < 8; ++j) {
        int sc, m;
        scale_min(j, sq, sc, m);
        const float dd = d * sc;
        if (dd == 0.0f)
            continue;
        const float dm = dmin * m;
        for (int i = 0; i < 32; ++i)
            L[32 * j + i] = uint8_t(std::clamp(nearest((x[32 * j + i] + dm) / dd), 0, nmax));
    }
}

void q4_k(const float* x, std::byte* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 256; ++b, x += 256, dst += 144) {
        uint8_t L[256];
        quantize_k(x, 15, dst, L);
        auto* q = reinterpret_cast<uint8_t*>(dst + 16);
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l)
                q[l] = uint8_t(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
    }
}

void q5_k(const float* x, std::byte* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 256; ++b, x += 256, dst += 176) {
        uint8_t L[256];
        quantize_k(x, 31, dst, L);
        auto* qh = reinterpret_cast<uint8_t*>(dst + 16);
        auto* ql = qh + 32;
        std::memset(qh, 0, 32);
        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < 256; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j], l2 = L[n + j + 32];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = uint8_t(l1 | (l2 << 4));
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }
}

// Q6_K block: 128 bytes of low nibbles, 64 of high bit pairs, 16 int8 scales (sub-blocks of 16), d (f16).
void q6_k(const float* x, std::byte* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 256; ++b, x += 256, dst += 210) {
        int8_t Ls[256];
        float scales[16];
        float max_scale = 0, max_abs = 0;
        for (int j = 0; j < 16; ++j) {
            scales[j] = fit_scale(16, 32, x + 16 * j, Ls + 16 * j);
            if (std::fabs(scales[j]) > max_abs) {
                max_abs = std::fabs(scales[j]);
                max_scale = scales[j];
            }
        }
        auto* ql = reinterpret_cast<uint8_t*>(dst);
        auto* qh = ql + 128;
        auto* sc = reinterpret_cast<int8_t*>(dst + 192);
        uint8_t L[256];
        if (max_abs == 0.0f) {
            std::memset(dst, 0, 210);
            continue;
        }
        const float iscale = -128.0f / max_scale;
        put_f16(dst + 208, 1.0f / iscale);
        const float d = get_f16(dst + 208);
        for (int j = 0; j < 16; ++j)
            sc[j] = int8_t(std::min(127, nearest(iscale * scales[j])));
        for (int j = 0; j < 16; ++j) {
            const float dd = d * sc[j];
            for (int i = 0; i < 16; ++i)
                L[16 * j + i] = dd == 0.0f ? 32 : uint8_t(std::clamp(nearest(x[16 * j + i] / dd), -32, 31) + 32);
        }
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l] & 0xF, q2 = L[j + l + 32] & 0xF, q3 = L[j + l + 64] & 0xF,
                              q4 = L[j + l + 96] & 0xF;
                ql[l] = uint8_t(q1 | (q3 << 4));
                ql[l + 32] = uint8_t(q2 | (q4 << 4));
                qh[l] = uint8_t((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) |
                                ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
    }
}

} // namespace

bool can_quantize(gguf::Type type)
{
    return type == gguf::Type::Q4_K || type == gguf::Type::Q5_K || type == gguf::Type::Q6_K;
}

void quantize_row(gguf::Type type, const float* src, std::byte* dst, uint64_t k)
{
    if (k % 256)
        throw std::runtime_error("quantize_row: k must be a multiple of 256");
    switch (type) {
    case gguf::Type::Q4_K: q4_k(src, dst, k); return;
    case gguf::Type::Q5_K: q5_k(src, dst, k); return;
    case gguf::Type::Q6_K: q6_k(src, dst, k); return;
    default: break;
    }
    throw std::runtime_error("no quantization to " + std::string(gguf::traits(type).name));
}

} // namespace mako::quant
