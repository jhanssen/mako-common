# Third-party per-format GEMV kernels (MIT), compiled offline into a metallib together with the slot-indexed wrapper
# that includes them.
set(GEMV_KERNELS_DIR ${CMAKE_BINARY_DIR}/third_party/gemv-kernels) # from cmake/llama_cpp_sources.cmake

set(GEMV_METALLIB ${CMAKE_BINARY_DIR}/gemv.metallib)
add_custom_command(
    OUTPUT ${GEMV_METALLIB}
    COMMAND xcrun -sdk macosx metal -std=metal4.0 -Werror -I ${GEMV_KERNELS_DIR} -I ${CMAKE_SOURCE_DIR}/src/kernels
            -c ${CMAKE_SOURCE_DIR}/src/kernels/gemv_slots.metal -o ${CMAKE_BINARY_DIR}/gemv.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/gemv.air -o ${GEMV_METALLIB}
    DEPENDS ${GEMV_KERNELS_DIR}/mul_mv.metal ${CMAKE_SOURCE_DIR}/src/kernels/gemv_slots.metal
            ${CMAKE_SOURCE_DIR}/src/kernels/moe_route.h
    COMMENT "Compiling reference GEMV kernels")
add_custom_target(gemv_metallib DEPENDS ${GEMV_METALLIB})
