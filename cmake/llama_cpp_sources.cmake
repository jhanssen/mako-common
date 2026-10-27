# Third-party sources (MIT) from one llama.cpp release: the per-format GEMV kernels and the Jinja engine for the GGUF
# chat templates, with the JSON library it wraps. The release's source archive is downloaded once and only these
# files are taken from it, each checked against its SHA-256 (the archive is generated on demand by the host, so its
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
    "common/jinja/lexer.h|93aecd72cc952fab08f657aa2a70e9f58421672b19d3179f8bdf555e683a1cc7|jinja/jinja"
    "common/jinja/lexer.cpp|ca5674297a3788d6893cf2e83812bce7e32354800ab65e34ee78d334a28892c1|jinja/jinja"
    "common/jinja/parser.h|6225eb759da910f3dd7753aed71c6938131cfa04c609d744e6f1390396422316|jinja/jinja"
    "common/jinja/parser.cpp|5a1da297892cf62e9e412c954e0a9e74401d0df13c7acc113887b6d081e8a62d|jinja/jinja"
    "common/jinja/runtime.h|c816c0e75d9b7f9f8e11c3280285c81086563026b02ed74767adf4448fbfbbab|jinja/jinja"
    "common/jinja/runtime.cpp|d846f033f16185d24b733233cf61aca296e312f8a08a7765bd38b30e963b8524|jinja/jinja"
    "common/jinja/string.h|d32ad2a319398b8dbc1cc43b39dd10ab2eb0ce2eae9d0eaf3c2162777dd615dc|jinja/jinja"
    "common/jinja/string.cpp|6b1b13dea3f94c4d542c15250f793abd638839733fd96a9da6704bfdab662e86|jinja/jinja"
    "common/jinja/value.h|6fae65f5eb4b1c07bf2e4a900799982fd40bba8d9db8333aba564463fca553b5|jinja/jinja"
    "common/jinja/value.cpp|a3272e014ec80951b6b5603750078c399f935c0c75175e43178ae7fee749f40c|jinja/jinja"
    "common/jinja/utils.h|61ae8bc57a7a551bdfb294a23a0ef8b5ba8e3cf4d695389f0f0797e2d7689781|jinja/jinja"
    "common/json.h|a352f31307f1be2bdb34c027b876f42a1c5d1ea775cc8e4f11d703b16c40d22f|jinja"
    "common/json.cpp|d8c556445cef8e7c93b0857f6ca7ae7ebd8295f169403ac66641b72f6b0598f2|jinja"
    "common/unicode.h|219b9257ac567a3e6bcf0f552c9a9ddc01d46a15eb4d2829ae5e069e643f0ec5|jinja"
    "common/unicode.cpp|191c00abab788e59e83e29fb877047fcbac1f4d6088c5e9962d38ac2e694686c|jinja"
    "vendor/nlohmann/json.hpp|08f19bbcc66ff3082f5caf58867b1232987dba98d82e2b29d870722fce42776b|jinja/nlohmann"
    "LICENSE|94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d|jinja")

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
