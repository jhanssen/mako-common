#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mako::text {

// SentencePiece's unigram model as a tokenizer.json describes it (T5's): the precompiled normalization map, runs of
// spaces made one, spaces as "▁" with one in front, then each piece split by the most likely segmentation; unknown
// characters become one unknown token per run. The normalization map is applied a character at a time, and to a
// character with its combining marks as a whole first (the reference takes whole grapheme clusters: text with other
// multi-character clusters can normalize differently).
class Unigram {
public:
    static std::expected<Unigram, std::string> load_json(const std::string& path);

    // Without the end token.
    std::vector<int32_t> encode(std::string_view text) const;
    int32_t eos() const { return eos_; }
    size_t size() const { return pieces_.size(); }

    // Exposed for tests.
    std::string normalize(std::string_view text) const;

private:
    // The normalization map's double-array trie: the offsets of the replacements of the keys that prefix `key`.
    std::vector<uint32_t> prefixes(std::string_view key) const;
    void segment(std::string_view piece, std::vector<int32_t>& out) const;

    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int32_t> ids_;
    size_t longest_ = 0; // bytes of the longest piece
    float unk_score_ = 0;
    int32_t unk_ = 0, eos_ = 1;
    std::vector<uint32_t> trie_;
    std::string replacements_; // NUL-terminated strings the trie's values point into
};

} // namespace mako::text
