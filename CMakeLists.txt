cmake_minimum_required(VERSION 4.3)
project(mako_common LANGUAGES CXX)

# What mako's programs share: GGUF and safetensors files, Metal (context, GEMV, GEMM, the kernel runtime), tokenizers,
# command-line options and presets, logging, HTTP access checks, and lending GPU memory between processes. Built on its
# own (with its tests) or added by a project with add_subdirectory.
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(FATAL_ERROR "mako targets Apple Silicon only")
endif()

set(MAKO_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${MAKO_COMMON_DIR}/cmake)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo)
    endif()
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

include(FetchContent)
FetchContent_Declare(metal_cpp
    URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
    URL_HASH SHA256=4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metal_cpp)
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
target_link_libraries(metal_cpp INTERFACE "-framework Foundation" "-framework Metal" "-framework QuartzCore")

include(llama_cpp_sources)
include(gemv_kernels)
include(server_deps)
include(mako_kernels)
# What mako_add_kernels() reads, for the project calling it.
if(NOT PROJECT_IS_TOP_LEVEL)
    set(MAKO_COMMON_DIR ${MAKO_COMMON_DIR} PARENT_SCOPE)
    set(GEMV_KERNELS_DIR ${GEMV_KERNELS_DIR} PARENT_SCOPE)
    set(MAKO_COMMON_METAL_SOURCES ${MAKO_COMMON_METAL_SOURCES} PARENT_SCOPE)
endif()

set(S ${MAKO_COMMON_DIR}/src)

add_library(mako_log INTERFACE)
target_include_directories(mako_log INTERFACE ${S})

add_library(mako_gguf STATIC ${S}/gguf/gguf.cpp ${S}/gguf/model_files.cpp ${S}/gguf/read_pool.cpp ${S}/quant/dequant.cpp
            ${S}/quant/quantize.cpp)
target_include_directories(mako_gguf PUBLIC ${S})

add_library(mako_mtl STATIC ${S}/mtl/metal_impl.cpp ${S}/mtl/context.cpp ${S}/mtl/gemv.cpp ${S}/mtl/gemm.cpp
            ${S}/mtl/embedded.cpp)
target_include_directories(mako_mtl PUBLIC ${S})
target_link_libraries(mako_mtl PUBLIC metal_cpp mako_gguf)
set_source_files_properties(${S}/mtl/embedded.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${CMAKE_BINARY_DIR};-Wno-c23-extensions"
    OBJECT_DEPENDS "${GEMV_METALLIB}")
add_dependencies(mako_mtl gemv_metallib)

# What any model's GPU code shares; nothing model-specific links into it.
add_library(mako_gpu STATIC ${S}/gpu/runtime.cpp)
target_include_directories(mako_gpu PUBLIC ${S})
target_link_libraries(mako_gpu PUBLIC mako_mtl mako_gguf)

add_library(mako_safetensors STATIC ${S}/safetensors/safetensors.cpp)
target_include_directories(mako_safetensors PUBLIC ${S})
target_link_libraries(mako_safetensors PUBLIC mako_gguf PRIVATE glaze_headers)

# The transformer models' common ground: weights from safetensors, attention heads and flash attention, LoRAs, and the
# Qwen text encoders the image and video models condition on.
add_library(mako_models STATIC ${S}/models/runtime.cpp ${S}/models/lora.cpp ${S}/models/qwen_text.cpp
            ${S}/models/conv.cpp ${S}/models/qwen_vision.cpp)
target_include_directories(mako_models PUBLIC ${S})
target_link_libraries(mako_models PUBLIC mako_gpu mako_safetensors mako_log PRIVATE glaze_headers)

add_library(mako_text STATIC ${S}/text/tokenizer.cpp ${S}/text/unigram.cpp)
target_include_directories(mako_text PUBLIC ${S})
target_link_libraries(mako_text PUBLIC mako_gguf PRIVATE glaze_headers)

add_library(mako_app STATIC ${S}/app/presets.cpp ${S}/app/cli.cpp)
target_include_directories(mako_app PUBLIC ${S})
target_link_libraries(mako_app PRIVATE glaze_headers)

add_library(mako_http STATIC ${S}/http/access.cpp)
target_include_directories(mako_http PUBLIC ${S})

add_library(mako_memory STATIC ${S}/memory/lender.cpp ${S}/memory/borrower.cpp)
target_include_directories(mako_memory PUBLIC ${S})
target_link_libraries(mako_memory PUBLIC mako_log PRIVATE glaze_headers httplib_headers)

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(bench)
endif()
