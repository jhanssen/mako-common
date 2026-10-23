#include "mtl/embedded.hpp"

namespace mako::mtl {

namespace {

constexpr unsigned char kKernels[] = {
#embed "mako.metallib"
};

constexpr unsigned char kGemv[] = {
#embed "gemv.metallib"
};

} // namespace

std::span<const std::byte> kernels_metallib() { return std::as_bytes(std::span(kKernels)); }
std::span<const std::byte> gemv_metallib() { return std::as_bytes(std::span(kGemv)); }

} // namespace mako::mtl
