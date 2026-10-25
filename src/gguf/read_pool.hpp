#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace mako::gguf {

// pread until `bytes` are in (throws std::runtime_error on an error or EOF).
void read_exact(int fd, void* dst, uint64_t bytes, uint64_t offset);

// Persistent threads for parallel reads: run() hands them tasks 0..n-1 and returns when all are done (rethrowing the
// first task's exception, if any).
class ReadPool {
public:
    explicit ReadPool(int threads);
    ~ReadPool();
    ReadPool(const ReadPool&) = delete;
    ReadPool& operator=(const ReadPool&) = delete;

    void run(size_t n, const std::function<void(size_t)>& task);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mako::gguf
