#include "quant/dequant.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mako::quant {

namespace {

float f16(const std::byte* p)
{
    _Float16 h;
    std::memcpy(&h, p, 2);
    return static_cast<float>(h);
}

float bf16(const std::byte* p)
{
    uint16_t u;
    std::memcpy(&u, p, 2);
    const uint32_t bits = uint32_t(u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void q8_0(const std::byte* src, float* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 32; ++b, src += 34) {
        const float d = f16(src);
        const auto* q = reinterpret_cast<const int8_t*>(src + 2);
        for (int i = 0; i < 32; ++i)
            *dst++ = d * q[i];
    }
}

void q6_k(const std::byte* src, float* dst, uint64_t k)
{
    for (uint64_t b = 0; b < k / 256; ++b, src += 210) {
        const auto* ql = reinterpret_cast<const uint8_t*>(src);
        const auto* qh = ql + 128;
        const auto* sc = reinterpret_cast<const int8_t*>(src + 192);
        const float d = f16(src + 208);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l] = d * sc[is] * q1;
                dst[l + 32] = d * sc[is + 2] * q2;
                dst[l + 64] = d * sc[is + 4] * q3;
                dst[l + 96] = d * sc[is + 6] * q4;
            }
            dst += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void scale_min_k4(int j, const uint8_t* q, int& sc, int& m)
{
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K (bits 4) and Q5_K (bits 5): super-blocks of 256 in 8 sub-blocks with 6-bit scales and mins.
void qk(const std::byte* src, float* dst, uint64_t k, int bits)
{
    const uint64_t bytes = bits == 4 ? 144 : 176;
    for (uint64_t b = 0; b < k / 256; ++b, src += bytes) {
        const float d = f16(src), dmin = f16(src + 2);
        const auto* sc = reinterpret_cast<const uint8_t*>(src + 4);
        const auto* qh = reinterpret_cast<const uint8_t*>(src + 16);
        const auto* ql = bits == 4 ? qh : qh + 32;
        for (int j = 0, is = 0; j < 256; j += 64, is += 2) {
            int s1, m1, s2, m2;
            scale_min_k4(is, sc, s1, m1);
            scale_min_k4(is + 1, sc, s2, m2);
            const uint8_t u1 = uint8_t(1 << is), u2 = uint8_t(2 << is);
            for (int l = 0; l < 32; ++l) {
                const int h1 = bits == 5 && (qh[l] & u1) ? 16 : 0, h2 = bits == 5 && (qh[l] & u2) ? 16 : 0;
                dst[l] = d * s1 * ((ql[l] & 0xF) + h1) - dmin * m1;
                dst[l + 32] = d * s2 * ((ql[l] >> 4) + h2) - dmin * m2;
            }
            dst += 64;
            ql += 32;
        }
    }
}

} // namespace

bool can_dequant(gguf::Type type)
{
    using gguf::Type;
    return type == Type::F32 || type == Type::F16 || type == Type::BF16 || type == Type::Q8_0 || type == Type::Q6_K ||
           type == Type::Q4_K || type == Type::Q5_K;
}

void dequant_row(gguf::Type type, const std::byte* src, float* dst, uint64_t k)
{
    using gguf::Type;
    switch (type) {
    case Type::F32: std::memcpy(dst, src, k * 4); return;
    case Type::F16:
        for (uint64_t i = 0; i < k; ++i)
            dst[i] = f16(src + 2 * i);
        return;
    case Type::BF16:
        for (uint64_t i = 0; i < k; ++i)
            dst[i] = bf16(src + 2 * i);
        return;
    case Type::Q8_0: q8_0(src, dst, k); return;
    case Type::Q6_K: q6_k(src, dst, k); return;
    case Type::Q4_K: qk(src, dst, k, 4); return;
    case Type::Q5_K: qk(src, dst, k, 5); return;
    default: break;
    }
    throw std::runtime_error("no CPU dequantization for " + std::string(gguf::traits(type).name));
}

} // namespace mako::quant
