#pragma once

#include "gguf/gguf.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mako::text {

// Byte-level BPE as described by a GGUF's tokenizer.ggml.* metadata (model "gpt2"; GLM's and DeepSeek-V3's
// pre-tokenizers), or by a Qwen tokenizer.json (no NFC normalization: text not already in NFC can split differently).
class Tokenizer {
public:
    enum class TokenType : int32_t { Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6 };

    static std::expected<Tokenizer, std::string> load(const gguf::File& meta);
    static std::expected<Tokenizer, std::string> load_json(const std::string& path);

    // Control tokens written in the text are recognized only with parse_special; user-defined ones always are.
    std::vector<int32_t> encode(std::string_view text, bool parse_special) const;
    // Control tokens are recognized except where they overlap one of the sorted, disjoint byte ranges [first, second)
    // (text from a chat template's input).
    std::vector<int32_t> encode(std::string_view text, std::span<const std::pair<size_t, size_t>> literal) const;
    // Control tokens are rendered as their text only with `special`.
    std::string decode(std::span<const int32_t> ids, bool special = true) const;

    int32_t find(std::string_view token_text) const;   // -1 if absent
    size_t size() const { return tokens_.size(); }
    int32_t eos() const { return eos_; }
    int32_t eot() const { return eot_; }

    // Exposed for tests: the pre-tokenizer's split of plain text.
    std::vector<std::string> split(std::string_view text) const;

private:
    void encode_plain(std::string_view text, std::vector<int32_t>& out) const;
    void bpe(const std::string& word, std::vector<int32_t>& out) const;

    std::vector<std::string> tokens_;
    std::vector<TokenType> types_;
    std::unordered_map<std::string, int32_t> ids_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
    std::vector<int32_t> specials_;                        // longest text first
    bool ignore_merges_ = false;
    bool deepseek_ = false; // DeepSeek-V3's pre-tokenizer, else GLM's
    uint32_t max_digits_ = 3; // GLM's pre-tokenizer: digits in runs of up to this many (Qwen's: 1)
    int32_t eos_ = -1, eot_ = -1;
};

} // namespace mako::text
