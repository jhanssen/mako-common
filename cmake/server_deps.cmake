# Header-only dependencies of the HTTP server, pinned by version and hash: glaze (JSON) and cpp-httplib.
FetchContent_Declare(glaze
    URL https://github.com/stephenberry/glaze/archive/refs/tags/v9.0.0.tar.gz
    URL_HASH SHA256=dd7b033c5bf6e4308615bb7834c541291d5b20f64121d9a68222a39414635517
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR no-cmake) # headers only; its own build is not configured
FetchContent_MakeAvailable(glaze)
add_library(glaze_headers INTERFACE)
target_include_directories(glaze_headers SYSTEM INTERFACE ${glaze_SOURCE_DIR}/include)

set(HTTPLIB_DIR ${CMAKE_BINARY_DIR}/third_party/httplib)
foreach(entry "httplib.h|e932d7771bdb93150868987e30112d9a31e24bef592d6d7a07e973d5845cf8d6"
              "LICENSE|4b45cbe16d7b71b89ae6127e26e0d90a029198ca5e958ad8e3d0b8bbed364d8b")
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 name)
    list(GET parts 1 hash)
    file(DOWNLOAD https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.60.1/${name} ${HTTPLIB_DIR}/${name}
         EXPECTED_HASH SHA256=${hash})
endforeach()
add_library(httplib_headers INTERFACE)
target_include_directories(httplib_headers SYSTEM INTERFACE ${HTTPLIB_DIR})
