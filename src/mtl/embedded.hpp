#pragma once

#include <cstddef>
#include <span>

namespace mako::mtl {

// Shader libraries compiled into the binary at build time.
std::span<const std::byte> kernels_metallib();
std::span<const std::byte> gemv_metallib();

} // namespace mako::mtl
