// Prints the per-simdgroup lane layout of a cooperative matmul2d result (bench/attn_ct.metal's coop probe), one
// line per (simdgroup, lane), elements as (col,row); rows held by each simdgroup summarized.
#include "mtl/context.hpp"
#include "mtl/embedded.hpp"

#include <print>
#include <set>
#include <string>

int main(int argc, char** argv)
{
    const std::string name = argc > 1 ? argv[1] : "coop_layout_probe_coop4_64x64";
    const int nsg = argc > 2 ? std::stoi(argv[2]) : 4;
    mako::mtl::Context ctx;
    auto lib = ctx.load(mako::mtl::kernels_metallib());
    auto pso = ctx.pipeline(lib.get(), name);
    mako::mtl::Ref<MTL::Buffer> out(ctx.device()->newBuffer(size_t(256) * 4096 * 3 * 4 + 16, MTL::ResourceStorageModeShared));
    auto* cb = ctx.queue()->commandBuffer();
    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso.get());
    enc->setBuffer(out.get(), 0, 0);
    enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32 * nsg, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    const int* o = static_cast<const int*>(out->contents());
    const int cap = o[128 * 4096 * 3];
    std::println("{}: capacity {} elements a lane", name, cap);
    for (int sg = 0; sg < nsg; ++sg) {
        std::set<int> rows;
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < cap; ++i) {
                const int* e = o + ((sg * 32 + lane) * cap + i) * 3;
                if (e[0])
                    rows.insert(e[2]);
            }
        std::string r;
        for (int x : rows)
            r += std::to_string(x) + " ";
        std::println("simdgroup {}: rows {}", sg, r);
        for (int lane : {0, 9, 23}) {
            std::string s;
            for (int i = 0; i < cap; ++i) {
                const int* e = o + ((sg * 32 + lane) * cap + i) * 3;
                s += e[0] ? std::format("({},{}) ", e[1], e[2]) : "- ";
            }
            std::println("  lane {:2}: {}...", lane, s);
        }
    }
    return 0;
}
