#pragma once

#include "gguf/gguf.hpp"

#include <cstddef>
#include <cstdint>

namespace mako::quant {

bool can_dequant(gguf::Type type);

// Dequantizes one row of k values (k a multiple of the type's block size).
void dequant_row(gguf::Type type, const std::byte* src, float* dst, uint64_t k);

} // namespace mako::quant
