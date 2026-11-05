#include "models/runtime.hpp"

#include "log/log.hpp"

#include <glaze/glaze.hpp>

#include <cstdlib>
#include <cstring>
#include <format>
#include <map>
#include <stdexcept>
#include <string_view>

namespace mako::models {

namespace {

struct LoraArgs {
    uint32_t k, rows, rank, b_rot;
    float scale;
};

} // namespace

uint32_t TransformerRuntime::add_lora(const std::string& path, float scale, std::optional<bool> merge)
{
    if (merge == false && !keeps_loras_unmerged())
        throw std::runtime_error(std::format("{}: {} keeps no LoRA unmerged", path, name()));
    auto file = safetensors::File::open(path);
    if (!file)
        throw std::runtime_error(file.error());
    if (const auto it = file->metadata().find("hayate_control");
        !control_lora_ && it != file->metadata().end() && it->second == "1")
        throw std::runtime_error(std::format("{}: a control adapter, not a LoRA (the model directory's \"control\")",
                                             path));
    // The adapter's alpha for every matrix, from a PEFT config in the metadata, or a plain "alpha" / "lora_alpha"
    // entry there.
    std::optional<double> alpha;
    if (const auto it = file->metadata().find("lora_adapter_metadata"); it != file->metadata().end()) {
        glz::generic j;
        if (!glz::read_json(j, it->second) && j.is_object())
            for (const char* k : {"transformer.lora_alpha", "lora_alpha"})
                if (j.contains(k) && j[k].is_number())
                    alpha = j[k].get_number();
    }
    for (const char* k : {"alpha", "lora_alpha"})
        if (const auto it = file->metadata().find(k); !alpha && it != file->metadata().end()) {
            char* end = nullptr;
            const double v = std::strtod(it->second.c_str(), &end);
            if (end != it->second.c_str() && *end == '\0' && v > 0)
                alpha = v;
        }
    struct Pair {
        const gguf::Tensor *a = nullptr, *b = nullptr, *alpha = nullptr;
    };
    std::map<std::string, Pair> mods;
    std::map<std::string, const gguf::Tensor*> bias_diffs, weight_diffs;
    std::vector<std::string> unknown;
    static constexpr std::pair<const char*, int> kSuffixes[] = {
        {".lora.down.weight", 0}, {".lora.up.weight", 1},  {".lora_down.weight", 0},       {".lora_up.weight", 1},
        {".lora_A.weight", 0},    {".lora_B.weight", 1},   {".lora_A.default.weight", 0}, {".lora_B.default.weight", 1},
        {".lora_down", 0},        {".lora_up", 1},         {".alpha", 2}};
    for (const auto& t : file->tensors()) {
        std::string_view name = t.name;
        for (const char* p : {"base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
                              "lora_unet_"})
            if (name.starts_with(p))
                name.remove_prefix(std::strlen(p));
        if (control_lora_ && name.starts_with("video_embedder."))
            continue; // the control adapter's input projection, which the model reads itself
        if (name.ends_with(".diff_b")) {
            bias_diffs[std::string(name.substr(0, name.size() - 7))] = &t;
            continue;
        }
        if (name.ends_with(".diff")) { // a whole weight's difference
            weight_diffs[std::string(name.substr(0, name.size() - 5))] = &t;
            continue;
        }
        bool known = false;
        for (const auto& [suffix, role] : kSuffixes)
            if (name.ends_with(suffix)) {
                Pair& p = mods[std::string(name.substr(0, name.size() - std::strlen(suffix)))];
                (role == 0 ? p.a : role == 1 ? p.b : p.alpha) = &t;
                known = true;
                break;
            }
        if (!known)
            unknown.push_back(t.name);
    }
    // Kohya's names: the module path with underscores for its dots; which underscores were dots, tried until one names
    // a weight.
    auto dotted = [&](const std::string& m, auto&& found) -> std::string {
        if (m.find('.') != std::string::npos || found(m))
            return m;
        std::vector<size_t> at;
        for (size_t i = 0; i < m.size(); ++i)
            if (m[i] == '_')
                at.push_back(i);
        if (at.size() > 16)
            return m;
        for (uint32_t bits = 1; bits < (1u << at.size()); ++bits) {
            std::string c = m;
            for (size_t i = 0; i < at.size(); ++i)
                if (bits >> i & 1)
                    c[at[i]] = '.';
            if (found(c))
                return c;
        }
        return m;
    };
    {
        std::map<std::string, Pair> renamed;
        for (auto& [m, p] : mods)
            renamed[dotted(m, [&](const std::string& c) {
                uint32_t r = 0, k = 0;
                return lora_target(c).w != nullptr || lora_cpu(c, r, k) != nullptr;
            })] = p;
        mods = std::move(renamed);
        std::map<std::string, const gguf::Tensor*> rb;
        for (auto& [m, t] : bias_diffs)
            rb[dotted(m, [&](const std::string& c) { return lora_bias(c) != nullptr; })] = t;
        bias_diffs = std::move(rb);
        std::map<std::string, const gguf::Tensor*> rw;
        for (auto& [m, t] : weight_diffs)
            rw[dotted(m, [&](const std::string& c) {
                uint32_t r = 0, k = 0;
                return lora_cpu(c, r, k) != nullptr;
            })] = t;
        weight_diffs = std::move(rw);
    }
    for (const auto& [m, p] : mods) {
        uint32_t r = 0, k = 0;
        if (!p.a || !p.b || (!lora_target(m).w && !lora_cpu(m, r, k)))
            unknown.push_back(m);
    }
    for (const auto& [m, t] : bias_diffs) {
        const Tensor* b = lora_bias(m);
        if (!b || b->type != gguf::Type::F32 || t->elements() != b->dims[0])
            unknown.push_back(m + ".diff_b");
    }
    for (const auto& [m, t] : weight_diffs) {
        uint32_t r = 0, k = 0;
        if (!lora_cpu(m, r, k) || t->elements() != uint64_t(r) * k)
            unknown.push_back(m + ".diff");
    }
    if (!unknown.empty())
        throw std::runtime_error(std::format("{}: {} keys {} does not adapt, e.g. {}", path, unknown.size(), name(),
                                             unknown.front()));
    uint32_t done = 0, rank_seen = 0, side = 0;
    for (const auto& [m, p] : mods) {
        const LoraTarget tg = lora_target(m);
        if (!tg.w) { // a small matrix on the CPU: W += s * B A there
            uint32_t rows = 0, k = 0;
            float* w = lora_cpu(m, rows, k);
            const uint32_t rank = uint32_t(p.a->dims[1]);
            if (p.a->dims[0] != k || p.b->dims[0] != rank || p.b->dims[1] != rows)
                throw std::runtime_error(std::format("{}: {} does not fit the weight it adapts", path, m));
            const double a = p.alpha ? read_floats(*file, *p.alpha).front() : alpha.value_or(double(rank));
            const auto av = read_floats(*file, *p.a), bv = read_floats(*file, *p.b);
            for (uint32_t i = 0; i < rows; ++i)
                for (uint32_t j = 0; j < k; ++j) {
                    double sum = 0;
                    for (uint32_t r = 0; r < rank; ++r)
                        sum += double(bv[uint64_t(i) * rank + r]) * av[uint64_t(r) * k + j];
                    w[uint64_t(i) * k + j] += float(double(scale) * a / double(rank) * sum);
                }
            ++done;
            continue;
        }
        const uint32_t k = uint32_t(tg.w->dims[0]), rank = uint32_t(p.a->dims[1]);
        // A is [rank][k] and B [rows][rank] (row-major), so their contiguous dimensions come first here.
        if (p.a->dims[0] != k || p.b->dims[0] != rank || p.b->dims[1] != tg.rows ||
            (tg.w->type != gguf::Type::BF16 && tg.w->type != gguf::Type::F16 && tg.w->type != gguf::Type::I8))
            throw std::runtime_error(std::format("{}: {} does not fit the weight it adapts", path, m));
        if (tg.w->type == gguf::Type::I8 && k > 16384)
            throw std::runtime_error(std::format("{}: {} has rows of {}, more than the int8 merge's 16384", path, m, k));
        double a = alpha.value_or(double(rank));
        if (p.alpha)
            a = read_floats(*file, *p.alpha).front();
        const auto av = read_floats(*file, *p.a), bv = read_floats(*file, *p.b);
        if (merge != true && lora_side(m, av, bv, rank, float(double(scale) * a / double(rank)))) {
            ++done;
            ++side;
            rank_seen = rank;
            continue;
        }
        MTL::Buffer* ab = scratch("l_a", av.size() * 4);
        MTL::Buffer* bb = scratch("l_b", bv.size() * 4);
        std::memcpy(ab->contents(), av.data(), av.size() * 4);
        std::memcpy(bb->contents(), bv.data(), bv.size() * 4);
        const LoraArgs la{k, tg.rows, rank, tg.b_rot, float(double(scale) * a / double(rank))};
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
        if (tg.w->type == gguf::Type::I8) { // a threadgroup per row: the row requantized to its new maximum
            enc->setComputePipelineState(pso("lora_merge_i8"));
            enc->setBuffer(tg.w->buf, tg.w->off + uint64_t(tg.row0) * k, 0);
            enc->setBuffer(ab, 0, 1);
            enc->setBuffer(bb, 0, 2);
            enc->setBytes(&la, sizeof(la), 3);
            enc->setBuffer(tg.w->buf, tg.w->off + uint64_t(tg.w->dims[1]) * k + uint64_t(tg.row0) * 4, 4);
            enc->dispatchThreadgroups(MTL::Size(tg.rows, 1, 1), MTL::Size(256, 1, 1));
        } else {
            enc->setComputePipelineState(pso(tg.w->type == gguf::Type::F16 ? "lora_merge_f16" : "lora_merge"));
            enc->setBuffer(tg.w->buf, tg.w->off + uint64_t(tg.row0) * k * 2, 0);
            enc->setBuffer(ab, 0, 1);
            enc->setBuffer(bb, 0, 2);
            enc->setBytes(&la, sizeof(la), 3);
            enc->dispatchThreads(MTL::Size(k, tg.rows, 1), MTL::Size(256, 1, 1));
        }
        enc->endEncoding();
        finish(cb);
        ++done;
        rank_seen = rank;
    }
    for (const auto& [m, t] : bias_diffs) {
        const Tensor* b = lora_bias(m);
        auto* dst = reinterpret_cast<float*>(static_cast<char*>(b->buf->contents()) + b->off);
        const auto d = read_floats(*file, *t);
        for (size_t i = 0; i < d.size(); ++i)
            dst[i] += scale * d[i];
        ++done;
    }
    for (const auto& [m, t] : weight_diffs) {
        uint32_t r = 0, k = 0;
        float* w = lora_cpu(m, r, k);
        const auto d = read_floats(*file, *t);
        for (size_t i = 0; i < d.size(); ++i)
            w[i] += scale * d[i];
        ++done;
    }
    if (!done)
        throw std::runtime_error(path + ": no LoRA weights in it");
    lora_merged();
    log::info("LoRA {}: {} tensors adapted ({} kept unmerged; rank {}, alpha {}), scale {}", path, done, side,
              rank_seen, alpha ? std::format("{}", *alpha) : std::string("= rank"), scale);
    return done;
}

} // namespace mako::models
