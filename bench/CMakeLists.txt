# Kernel benchmarks (the kernels come from tests' common_test_kernels, linked into mako_mtl).
add_executable(attn_bench attn_bench.cpp)
target_link_libraries(attn_bench PRIVATE mako_mtl)
add_executable(dense_gemm_bench dense_gemm_bench.cpp)
target_link_libraries(dense_gemm_bench PRIVATE mako_mtl)
