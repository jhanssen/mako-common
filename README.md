# mako-common

The code [mako](https://github.com/jhanssen/mako) (LLM inference) and
[mako-image](https://github.com/jhanssen/mako-image) (image generation) share, added to each as a git submodule
(`external/mako-common`) and built with `add_subdirectory`:

- `gguf`, `quant`, `safetensors`: model files and quantized formats
- `mtl`, `gpu`: Metal context, GEMV and GEMM, the kernel runtime; `src/kernels` the kernels every program has
- `text`: the BPE and unigram tokenizers
- `app`: command-line options and presets; `log`: timestamped logging
- `http`: API keys and loopback checks for the HTTP servers
- `memory`: lending GPU memory between two processes on one machine (`memory/lender.hpp` describes the protocol)

A program builds its kernels with `mako_add_kernels(<target> SOURCES <its .metal files>)` (`cmake/mako_kernels.cmake`):
one metallib of the common kernels and its own, linked as the object library `<target>`. Third-party llama.cpp files
a project needs besides the GEMV kernels go in `MAKO_LLAMA_CPP_EXTRA_FILES` before `add_subdirectory`.

## Build and test

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build -L cpu      # no GPU needed
ctest --test-dir build -L kernels  # the GPU kernels
```

Requirements: Apple Silicon, macOS 26 or later, Xcode with the Metal toolchain, CMake 4.3 or later, Ninja.
