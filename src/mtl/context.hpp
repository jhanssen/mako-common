#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <span>
#include <string>
#include <string_view>

namespace mako::mtl {

// Owning handle for metal-cpp objects returned by new*/copy* calls.
template <typename T>
class Ref {
public:
    Ref() = default;
    explicit Ref(T* p) : p_(p) {}
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Ref& operator=(Ref&& o) noexcept
    {
        if (this != &o) {
            reset();
            p_ = o.p_;
            o.p_ = nullptr;
        }
        return *this;
    }
    ~Ref() { reset(); }

    void reset()
    {
        if (p_)
            p_->release();
        p_ = nullptr;
    }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }

private:
    T* p_ = nullptr;
};

struct DeviceInfo {
    std::string name;
    bool apple9 = false;
    bool apple10 = false;
    bool metal4 = false;
    uint64_t max_buffer = 0;
    uint64_t recommended_working_set = 0;
    uint64_t max_threadgroup_memory = 0;
    // GPU timestamp sampling: where it can be taken, and whether the timestamp counter set exists.
    bool counters_stage = false, counters_dispatch = false, counters_blit = false, timestamp_set = false;
};

class Context {
public:
    Context();

    MTL::Device* device() const { return device_.get(); }
    MTL::CommandQueue* queue() const { return queue_.get(); }
    const DeviceInfo& info() const { return info_; }

    // Throws std::runtime_error with the compiler log on failure.
    Ref<MTL::Library> compile(std::string_view source) const;
    Ref<MTL::Library> load(std::span<const std::byte> metallib) const;
    Ref<MTL::ComputePipelineState> pipeline(MTL::Library* lib, std::string_view fn,
                                            const MTL::FunctionConstantValues* constants = nullptr) const;

private:
    Ref<MTL::Device> device_;
    Ref<MTL::CommandQueue> queue_;
    DeviceInfo info_;
};

} // namespace mako::mtl
