#include "mtl/embedded.hpp"

namespace mako::mtl {

namespace {

constexpr unsigned char kGemv[] = {
#embed "gemv.metallib"
};

} // namespace

std::span<const std::byte> gemv_metallib() { return std::as_bytes(std::span(kGemv)); }

} // namespace mako::mtl
