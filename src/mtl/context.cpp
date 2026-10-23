#include "mtl/context.hpp"

#include <stdexcept>

namespace mako::mtl {

namespace {

NS::String* ns_str(std::string_view s)
{
    return NS::String::alloc()->init(const_cast<char*>(s.data()), s.size(), NS::UTF8StringEncoding, false)->autorelease();
}

std::string error_text(NS::Error* err)
{
    if (!err)
        return "unknown error";
    return err->localizedDescription()->utf8String();
}

} // namespace

Context::Context()
{
    device_ = Ref<MTL::Device>(MTL::CreateSystemDefaultDevice());
    if (!device_)
        throw std::runtime_error("no Metal device");
    queue_ = Ref<MTL::CommandQueue>(device_->newCommandQueue());

    info_.name = device_->name()->utf8String();
    info_.apple9 = device_->supportsFamily(MTL::GPUFamilyApple9);
    info_.apple10 = device_->supportsFamily(MTL::GPUFamilyApple10);
    info_.metal4 = device_->supportsFamily(MTL::GPUFamilyMetal4);
    info_.max_buffer = device_->maxBufferLength();
    info_.recommended_working_set = device_->recommendedMaxWorkingSetSize();
    info_.max_threadgroup_memory = device_->maxThreadgroupMemoryLength();
}

Ref<MTL::Library> Context::compile(std::string_view source) const
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    Ref<MTL::CompileOptions> opts(MTL::CompileOptions::alloc()->init());
    opts->setLanguageVersion(MTL::LanguageVersion4_0);
    opts->setMathMode(MTL::MathModeSafe);
    NS::Error* err = nullptr;
    Ref<MTL::Library> lib(device_->newLibrary(ns_str(source), opts.get(), &err));
    if (!lib) {
        std::string msg = "Metal compile failed: " + error_text(err);
        pool->release();
        throw std::runtime_error(msg);
    }
    pool->release();
    return lib;
}

Ref<MTL::ComputePipelineState> Context::pipeline(MTL::Library* lib, std::string_view fn) const
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    Ref<MTL::Function> f(lib->newFunction(ns_str(fn)));
    if (!f) {
        pool->release();
        throw std::runtime_error("no Metal function " + std::string(fn));
    }
    NS::Error* err = nullptr;
    Ref<MTL::ComputePipelineState> p(device_->newComputePipelineState(f.get(), &err));
    if (!p) {
        std::string msg = "pipeline " + std::string(fn) + ": " + error_text(err);
        pool->release();
        throw std::runtime_error(msg);
    }
    pool->release();
    return p;
}

} // namespace mako::mtl
