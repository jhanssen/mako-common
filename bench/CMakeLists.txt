# Kernel benchmarks (the kernels come from tests' common_test_kernels, linked into mako_mtl; it also compiles
# attn_ct.metal, the attention experiments).
add_executable(attn_bench attn_bench.cpp)
target_link_libraries(attn_bench PRIVATE mako_mtl)
add_executable(dense_gemm_bench dense_gemm_bench.cpp)
target_link_libraries(dense_gemm_bench PRIVATE mako_mtl)
add_executable(coop_probe coop_probe.cpp)
target_link_libraries(coop_probe PRIVATE mako_mtl)
add_executable(mm_rate_bench mm_rate_bench.cpp)
target_link_libraries(mm_rate_bench PRIVATE mako_mtl)
add_executable(coop_probe4 coop_probe4.cpp)
target_link_libraries(coop_probe4 PRIVATE mako_mtl)
