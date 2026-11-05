// Prints the lane layout of matmul2d's cooperative result tensor (bench/attn_ct.metal's probe).
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <print>
#include <string>

int main(int argc, char** argv)
{
    const std::string name = argc > 1 ? argv[1] : "coop_layout_probe_16x64";
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto pso = ctx.pipeline(lib.get(), name);
    mako::mtl::Ref<MTL::Buffer> out(ctx.device()->newBuffer(32 * 4096 * 3 * 4 + 16, MTL::ResourceStorageModeShared));
    auto* cb = ctx.queue()->commandBuffer();
    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso.get());
    enc->setBuffer(out.get(), 0, 0);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    const int* o = static_cast<const int*>(out->contents());
    const int cap = o[32 * 4096 * 3];
    std::println("{}: capacity {} elements a lane", name, cap);
    for (int lane = 0; lane < 32; ++lane) {
        std::string s;
        for (int i = 0; i < cap; ++i) {
            const int* e = o + (lane * cap + i) * 3;
            s += e[0] ? std::format("({},{}) ", e[1], e[2]) : "- ";
        }
        std::println("lane {:2}: {}", lane, s);
    }
    return 0;
}
