# ctest -L cpu: no GPU and no model (what CI runs); -L kernels: the GPU kernels against references.
mako_add_kernels(common_test_kernels SOURCES ${PROJECT_SOURCE_DIR}/bench/attn_ct.metal)

add_executable(quant_test quant_test.cpp)
target_link_libraries(quant_test PRIVATE mako_mtl mako_gguf)
add_test(NAME quant COMMAND quant_test)

add_executable(gemv_own_test gemv_own_test.cpp)
target_link_libraries(gemv_own_test PRIVATE mako_mtl)
add_test(NAME gemv_own COMMAND gemv_own_test)
add_executable(gemv_rows_test gemv_rows_test.cpp)
target_link_libraries(gemv_rows_test PRIVATE mako_mtl)
add_test(NAME gemv_rows COMMAND gemv_rows_test)
set_tests_properties(quant gemv_own gemv_rows PROPERTIES LABELS kernels)

add_executable(presets_test presets_test.cpp)
target_link_libraries(presets_test PRIVATE mako_app)
add_test(NAME presets COMMAND presets_test ${CMAKE_CURRENT_BINARY_DIR})
add_executable(cli_test cli_test.cpp)
target_link_libraries(cli_test PRIVATE mako_app)
add_test(NAME cli COMMAND cli_test ${CMAKE_CURRENT_BINARY_DIR})

add_executable(http_test http_test.cpp)
target_link_libraries(http_test PRIVATE mako_http)
add_test(NAME http COMMAND http_test)

# Lending memory between two processes: a lender and a borrower over HTTP on loopback.
add_executable(memory_test memory_test.cpp)
target_link_libraries(memory_test PRIVATE mako_memory httplib_headers)
add_test(NAME memory COMMAND memory_test)
set_tests_properties(presets cli http memory PROPERTIES LABELS cpu)
