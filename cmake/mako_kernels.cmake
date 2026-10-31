# mako_add_kernels(<target> [SOURCES <file.metal>...] [DEPENDS <file>...]): one metallib of the common kernels and
# these, compiled with warnings as errors, and an object library <target> defining mako::mtl::kernels_metallib() with it
# embedded. A program links exactly one such library.
set(MAKO_COMMON_METAL_SOURCES ${MAKO_COMMON_DIR}/src/kernels/gemm.metal ${MAKO_COMMON_DIR}/src/kernels/gemv.metal
    ${MAKO_COMMON_DIR}/src/kernels/ops.metal)

function(mako_add_kernels target)
    cmake_parse_arguments(K "" "" "SOURCES;DEPENDS" ${ARGN})
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/${target})
    set(airs)
    foreach(src ${MAKO_COMMON_METAL_SOURCES} ${K_SOURCES})
        get_filename_component(stem ${src} NAME_WE)
        set(air ${dir}/${stem}.air)
        add_custom_command(OUTPUT ${air}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
            COMMAND xcrun -sdk macosx metal -std=metal4.0 -Wall -Werror -I ${GEMV_KERNELS_DIR}
                    -I ${MAKO_COMMON_DIR}/src/kernels -c ${src} -o ${air}
            DEPENDS ${src} ${GEMV_KERNELS_DIR}/ggml-common.h ${MAKO_COMMON_DIR}/src/kernels/moe_route.h ${K_DEPENDS}
            COMMENT "Compiling ${stem}.metal")
        list(APPEND airs ${air})
    endforeach()
    set(lib ${dir}/kernels.metallib)
    add_custom_command(OUTPUT ${lib}
        COMMAND xcrun -sdk macosx metallib ${airs} -o ${lib}
        DEPENDS ${airs}
        COMMENT "Linking ${target}'s metallib")
    add_custom_target(${target}_metallib DEPENDS ${lib})
    set(src ${dir}/kernels_metallib.cpp)
    file(WRITE ${src}.in
        "#include \"mtl/embedded.hpp\"\n\nnamespace mako::mtl {\n\nnamespace {\n\nconstexpr unsigned char kKernels[] = {\n"
        "#embed \"kernels.metallib\"\n};\n\n} // namespace\n\n"
        "std::span<const std::byte> kernels_metallib() { return std::as_bytes(std::span(kKernels)); }\n\n"
        "} // namespace mako::mtl\n")
    configure_file(${src}.in ${src} COPYONLY)
    add_library(${target} OBJECT ${src})
    target_link_libraries(${target} PUBLIC mako_mtl)
    set_source_files_properties(${src} PROPERTIES
        COMPILE_OPTIONS "--embed-dir=${dir};-Wno-c23-extensions"
        OBJECT_DEPENDS ${lib})
    add_dependencies(${target} ${target}_metallib)
endfunction()
