#pragma once

#include "gguf/gguf.hpp"

#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mako::gguf {

// Reads an architecture's keys (<arch>.<key>) from a GGUF's metadata, collecting the names of missing ones in
// `missing` instead of failing at the first.
class Keys {
public:
    Keys(const File& f, std::string arch) : f_(f), prefix_(std::move(arch) + ".") {}

    std::string missing;

    uint32_t u32(std::string_view key)
    {
        auto v = f_.get_int(full(key));
        if (!v) {
            note(key);
            return 0;
        }
        return static_cast<uint32_t>(*v);
    }

    float f32(std::string_view key, std::optional<float> fallback = std::nullopt)
    {
        auto v = f_.get_float(full(key));
        if (!v) {
            if (fallback)
                return *fallback;
            note(key);
            return 0.0f;
        }
        return static_cast<float>(*v);
    }

    bool flag(std::string_view key, bool fallback)
    {
        const Value* v = f_.find(full(key));
        if (auto* b = v ? std::get_if<bool>(v) : nullptr)
            return *b;
        return fallback;
    }

    // A per-layer array, or one scalar broadcast to every layer.
    std::vector<double> per_layer(std::string_view key, uint32_t n)
    {
        const Value* v = f_.find(full(key));
        std::vector<double> out;
        if (!v) {
            note(key);
            return std::vector<double>(n, 0.0);
        }
        auto num = [](const Value& x) -> double {
            if (auto* u = std::get_if<uint64_t>(&x))
                return double(*u);
            if (auto* i = std::get_if<int64_t>(&x))
                return double(*i);
            if (auto* d = std::get_if<double>(&x))
                return *d;
            return 0.0;
        };
        if (auto* a = std::get_if<Array>(v)) {
            for (const auto& item : a->items)
                out.push_back(num(item));
            if (out.size() != n)
                missing += std::format(" {}(length {} != {})", key, out.size(), n);
            out.resize(n, 0.0);
        } else {
            out.assign(n, num(*v));
        }
        return out;
    }

    // An integer array (empty, noted missing, when absent; required: false leaves it unnoted).
    std::vector<int64_t> ints(std::string_view key, bool required = true)
    {
        const Value* v = f_.find(full(key));
        std::vector<int64_t> out;
        const auto* a = v ? std::get_if<Array>(v) : nullptr;
        if (!a) {
            if (required)
                note(key);
            return out;
        }
        for (const auto& item : a->items) {
            if (auto* u = std::get_if<uint64_t>(&item))
                out.push_back(int64_t(*u));
            else if (auto* i = std::get_if<int64_t>(&item))
                out.push_back(*i);
            else
                missing += std::format(" {}(not integers)", key);
        }
        return out;
    }

    bool has(std::string_view key) const { return f_.find(full(key)) != nullptr; }
    // `key`, or `other` when only that one is present (two names for the same value).
    std::string_view either(std::string_view key, std::string_view other) const { return has(key) || !has(other) ? key : other; }

private:
    std::string full(std::string_view key) const { return prefix_ + std::string(key); }
    void note(std::string_view key) { missing += " " + std::string(key); }

    const File& f_;
    std::string prefix_;
};

} // namespace mako::gguf
