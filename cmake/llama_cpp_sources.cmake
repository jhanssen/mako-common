# Third-party sources (MIT) from one llama.cpp release: the per-format GEMV kernels, and whatever entries a project
# adds in MAKO_LLAMA_CPP_EXTRA_FILES (set before mako-common is added). The release's source archive is downloaded once
# and only these files are taken from it, each checked against its SHA-256 (the archive is generated on demand by the host, so its
# contents are what is pinned, not its bytes). Each entry: path in the release | SHA-256 | destination under
# third_party/.
set(LLAMA_CPP_RELEASE b11869)
set(_llama_url https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_CPP_RELEASE}.tar.gz)
set(_third_party ${CMAKE_BINARY_DIR}/third_party)
set(_llama_files
    "ggml/src/ggml-metal/kernels/mul_mv.metal|3b6250aa1ec74091bbadedbae35355c2d6ea5d566c2033f91d9aecd896c69059|gemv-kernels"
    "ggml/src/ggml-metal/kernels/common.h|525d4d7f9dbdaef3702195be92235ba2cc7cc86d5c97101d374a52b06117215e|gemv-kernels"
    "ggml/src/ggml-metal/kernels/dequantize.h|b64da60146cad11863bae09873e542dce76897c787849d47abaa2a5837d5bf8f|gemv-kernels"
    "ggml/src/ggml-metal/ggml-metal-impl.h|87925f2328dba82ab5e22cb67ae562fce843b4e04debf95d09443920da93d3de|gemv-kernels"
    "ggml/src/ggml-common.h|0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c|gemv-kernels"
    "LICENSE|94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d|gemv-kernels"
    ${MAKO_LLAMA_CPP_EXTRA_FILES})

# Whether every file is in place with its hash (then nothing is downloaded).
set(_llama_missing FALSE)
foreach(entry ${_llama_files})
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 path)
    list(GET parts 1 hash)
    list(GET parts 2 dest)
    get_filename_component(name ${path} NAME)
    set(_file ${_third_party}/${dest}/${name})
    if(EXISTS ${_file})
        file(SHA256 ${_file} _have)
    else()
        set(_have "")
    endif()
    if(NOT _have STREQUAL hash)
        set(_llama_missing TRUE)
    endif()
endforeach()

if(_llama_missing)
    set(_archive ${_third_party}/llama.cpp-${LLAMA_CPP_RELEASE}.tar.gz)
    set(_extract ${_third_party}/llama.cpp-${LLAMA_CPP_RELEASE})
    if(NOT EXISTS ${_archive})
        message(STATUS "Downloading llama.cpp ${LLAMA_CPP_RELEASE} sources")
        file(DOWNLOAD ${_llama_url} ${_archive}.part STATUS _status TLS_VERIFY ON)
        list(GET _status 0 _code)
        if(NOT _code EQUAL 0)
            file(REMOVE ${_archive}.part)
            list(GET _status 1 _msg)
            message(FATAL_ERROR "cannot download ${_llama_url}: ${_msg}")
        endif()
        file(RENAME ${_archive}.part ${_archive})
    endif()
    set(_patterns "")
    foreach(entry ${_llama_files})
        string(REPLACE "|" ";" parts ${entry})
        list(GET parts 0 path)
        list(APPEND _patterns "llama.cpp-${LLAMA_CPP_RELEASE}/${path}")
    endforeach()
    file(REMOVE_RECURSE ${_extract})
    file(ARCHIVE_EXTRACT INPUT ${_archive} DESTINATION ${_third_party} PATTERNS ${_patterns})
    foreach(entry ${_llama_files})
        string(REPLACE "|" ";" parts ${entry})
        list(GET parts 0 path)
        list(GET parts 1 hash)
        list(GET parts 2 dest)
        get_filename_component(name ${path} NAME)
        set(_src ${_extract}/${path})
        if(NOT EXISTS ${_src})
            message(FATAL_ERROR "llama.cpp ${LLAMA_CPP_RELEASE}: ${path} missing from ${_archive}")
        endif()
        file(SHA256 ${_src} _have)
        if(NOT _have STREQUAL hash)
            message(FATAL_ERROR "llama.cpp ${LLAMA_CPP_RELEASE}: ${path} has SHA-256 ${_have}, expected ${hash}")
        endif()
        file(MAKE_DIRECTORY ${_third_party}/${dest})
        file(COPY_FILE ${_src} ${_third_party}/${dest}/${name} ONLY_IF_DIFFERENT)
    endforeach()
    file(REMOVE_RECURSE ${_extract})
    file(REMOVE ${_archive})
endif()
