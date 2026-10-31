#pragma once

#include <cstddef>
#include <span>

namespace mako::mtl {

// Shader libraries compiled into the binary at build time. kernels_metallib(): the common kernels and the program's
// own, defined by the library mako_add_kernels() makes (cmake/mako_kernels.cmake).
std::span<const std::byte> kernels_metallib();
std::span<const std::byte> gemv_metallib();

} // namespace mako::mtl
