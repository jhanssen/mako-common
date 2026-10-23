#include "gguf/model_files.hpp"

#include <format>
#include <regex>

namespace mako::gguf {

std::expected<ModelFiles, std::string> ModelFiles::open(const std::string& first_shard)
{
    ModelFiles m;
    auto first = File::open(first_shard);
    if (!first)
        return std::unexpected(first.error());
    m.files_.push_back(std::move(*first));

    const int64_t count = m.meta().get_int("split.count").value_or(1);
    if (count <= 1)
        return m;

    static const std::regex shard_re(R"((.*-)(\d{5})(-of-)(\d{5})(\.gguf)$)");
    std::smatch sm;
    if (!std::regex_match(first_shard, sm, shard_re) || std::stoi(sm[2]) != 1)
        return std::unexpected(std::format("{}: split.count is {} but the name is not shard 00001", first_shard, count));

    for (int64_t i = 2; i <= count; ++i) {
        const std::string path = std::format("{}{:05}{}{}{}", sm[1].str(), i, sm[3].str(), sm[4].str(), sm[5].str());
        auto f = File::open(path);
        if (!f)
            return std::unexpected(f.error());
        if (f->get_int("split.no").value_or(-1) != i - 1)
            return std::unexpected(std::format("{}: unexpected split.no", path));
        m.files_.push_back(std::move(*f));
    }

    if (auto expect = m.meta().get_int("split.tensors.count"); expect && uint64_t(*expect) != m.tensor_count())
        return std::unexpected(std::format("{}: shards hold {} tensors, split.tensors.count is {}", first_shard,
                                           m.tensor_count(), *expect));
    return m;
}

ModelFiles::Ref ModelFiles::find(std::string_view name) const
{
    for (const auto& f : files_)
        if (const Tensor* t = f.tensor(name))
            return {&f, t};
    return {};
}

uint64_t ModelFiles::tensor_count() const
{
    uint64_t n = 0;
    for (const auto& f : files_)
        n += f.tensors().size();
    return n;
}

} // namespace mako::gguf
