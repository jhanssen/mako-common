# Third-party per-format GEMV kernels (MIT), pinned by commit and hash, compiled offline into a metallib together
# with the slot-indexed wrapper that includes them.
set(GEMV_KERNELS_COMMIT 9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e)
set(GEMV_KERNELS_DIR ${CMAKE_BINARY_DIR}/third_party/gemv-kernels)
set(_base https://raw.githubusercontent.com/ggml-org/llama.cpp/${GEMV_KERNELS_COMMIT})
set(_files
    "ggml/src/ggml-metal/kernels/mul_mv.metal|8fead8d2bc076f241cb101b2fbeb7b4cc7203aba72ffd9f1b5e297b16cd45a7c"
    "ggml/src/ggml-metal/kernels/common.h|525d4d7f9dbdaef3702195be92235ba2cc7cc86d5c97101d374a52b06117215e"
    "ggml/src/ggml-metal/kernels/dequantize.h|b64da60146cad11863bae09873e542dce76897c787849d47abaa2a5837d5bf8f"
    "ggml/src/ggml-metal/ggml-metal-impl.h|9594522f86ac135e8faa32d124d24d809f1c012b38a825ce7b3680a0962504d9"
    "ggml/src/ggml-common.h|0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c"
    "LICENSE|94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d")
foreach(entry ${_files})
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 path)
    list(GET parts 1 hash)
    get_filename_component(name ${path} NAME)
    file(DOWNLOAD ${_base}/${path} ${GEMV_KERNELS_DIR}/${name} EXPECTED_HASH SHA256=${hash})
endforeach()

set(GEMV_METALLIB ${CMAKE_BINARY_DIR}/gemv.metallib)
add_custom_command(
    OUTPUT ${GEMV_METALLIB}
    COMMAND xcrun -sdk macosx metal -std=metal4.0 -I ${GEMV_KERNELS_DIR} -I ${CMAKE_SOURCE_DIR}/src/kernels
            -c ${CMAKE_SOURCE_DIR}/src/kernels/gemv_slots.metal -o ${CMAKE_BINARY_DIR}/gemv.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/gemv.air -o ${GEMV_METALLIB}
    DEPENDS ${GEMV_KERNELS_DIR}/mul_mv.metal ${CMAKE_SOURCE_DIR}/src/kernels/gemv_slots.metal
            ${CMAKE_SOURCE_DIR}/src/kernels/moe_route.h
    COMMENT "Compiling reference GEMV kernels")
add_custom_target(gemv_metallib DEPENDS ${GEMV_METALLIB})
