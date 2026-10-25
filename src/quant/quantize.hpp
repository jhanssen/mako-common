#pragma once

#include "gguf/gguf.hpp"

#include <cstddef>
#include <cstdint>

namespace mako::quant {

// Q4_K, Q5_K and Q6_K (super-blocks of 256).
bool can_quantize(gguf::Type type);

// Quantizes one row of k floats (k a multiple of 256) into dst (k / 256 blocks of the type).
void quantize_row(gguf::Type type, const float* src, std::byte* dst, uint64_t k);

} // namespace mako::quant
