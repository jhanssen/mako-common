#include "text/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <variant>

namespace mako::text {

namespace {

struct CodepointRange {
    uint32_t first, last;
};

#include "text/unicode_tables.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&r)[N], uint32_t cp)
{
    auto it = std::upper_bound(std::begin(r), std::end(r), cp, [](uint32_t c, const CodepointRange& x) { return c < x.first; });
    return it != std::begin(r) && cp <= std::prev(it)->last;
}

bool is_letter(uint32_t cp) { return in_ranges(kLetterRanges, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumberRanges, cp); }

// The Unicode White_Space property.
bool is_space(uint32_t cp)
{
    switch (cp) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0x85: case 0xA0: case 0x1680:
    case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
        return true;
    default:
        return cp >= 0x2000 && cp <= 0x200A;
    }
}

bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Decodes UTF-8; each byte of an invalid sequence becomes U+FFFD. offs[i] is the byte offset of cps[i].
void decode_utf8(std::string_view s, std::vector<uint32_t>& cps, std::vector<size_t>& offs)
{
    size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        auto cont = [&](size_t k) { return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80; };
        if (b < 0x80) {
            cp = b;
        } else if ((b & 0xE0) == 0xC0 && cont(1)) {
            cp = (uint32_t(b & 0x1F) << 6) | (s[i + 1] & 0x3F);
            len = cp >= 0x80 ? 2 : 1;
        } else if ((b & 0xF0) == 0xE0 && cont(1) && cont(2)) {
            cp = (uint32_t(b & 0x0F) << 12) | (uint32_t(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            len = cp >= 0x800 ? 3 : 1;
        } else if ((b & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
            cp = (uint32_t(b & 0x07) << 18) | (uint32_t(s[i + 1] & 0x3F) << 12) | (uint32_t(s[i + 2] & 0x3F) << 6) |
                 (s[i + 3] & 0x3F);
            len = cp >= 0x10000 && cp <= 0x10FFFF ? 4 : 1;
        }
        if (len == 1 && b >= 0x80)
            cp = 0xFFFD;
        cps.push_back(cp);
        offs.push_back(i);
        i += len;
    }
    offs.push_back(s.size());
}

void append_utf8(std::string& out, uint32_t cp)
{
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xC0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += char(0xE0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | (cp >> 18));
        out += char(0x80 | ((cp >> 12) & 0x3F));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
}

// The byte-level BPE alphabet: printable bytes map to themselves, the rest to U+0100 and up.
struct ByteMap {
    std::array<std::string, 256> to_text;
    std::unordered_map<uint32_t, uint8_t> to_byte;

    ByteMap()
    {
        uint32_t next = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
            const uint32_t cp = printable ? b : next++;
            append_utf8(to_text[b], cp);
            to_byte[cp] = uint8_t(b);
        }
    }
};

const ByteMap& byte_map()
{
    static const ByteMap m;
    return m;
}

// Lengths of the pre-tokenizer's pieces, in code points:
//   's|'t|'re|'ve|'m|'ll|'d (any case) | [^\r\n\pL\pN]?\pL+ | \pN{1,3} | ' '?[^\s\pL\pN]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<size_t> split_lengths(const std::vector<uint32_t>& c)
{
    std::vector<size_t> out;
    const size_t n = c.size();
    auto lower = [&](size_t i) { return i < n && c[i] < 0x80 ? uint32_t(c[i] | 0x20) : 0u; };
    auto letter = [&](size_t i) { return i < n && is_letter(c[i]); };
    auto number = [&](size_t i) { return i < n && is_number(c[i]); };
    auto space = [&](size_t i) { return i < n && is_space(c[i]); };
    auto other = [&](size_t i) { return i < n && !is_space(c[i]) && !is_letter(c[i]) && !is_number(c[i]); };

    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        if (c[i] == '\'') {
            const uint32_t a = lower(i + 1), b = lower(i + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd')
                len = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                len = 3;
        }
        if (!len) {
            size_t j = i;
            if (!letter(j) && !number(j) && !is_newline(c[j]) && letter(j + 1))
                ++j;
            if (letter(j)) {
                while (letter(j))
                    ++j;
                len = j - i;
            }
        }
        if (!len && number(i)) {
            size_t j = i;
            while (j < i + 3 && number(j))
                ++j;
            len = j - i;
        }
        if (!len) {
            size_t j = i;
            if (c[j] == ' ' && other(j + 1))
                ++j;
            if (other(j)) {
                while (other(j))
                    ++j;
                while (j < n && is_newline(c[j]))
                    ++j;
                len = j - i;
            }
        }
        if (!len && space(i)) {
            size_t end = i;
            while (space(end))
                ++end;
            size_t last_nl = n;
            for (size_t j = i; j < end; ++j)
                if (is_newline(c[j]))
                    last_nl = j;
            if (last_nl != n)
                len = last_nl + 1 - i;
            else if (end == n || end - i >= 2)
                len = end == n ? end - i : end - i - 1;
            else
                len = 1;
        }
        if (!len)
            len = 1;
        out.push_back(len);
        i += len;
    }
    return out;
}

const gguf::Array* array_of(const gguf::File& f, std::string_view key)
{
    const gguf::Value* v = f.find(key);
    return v ? std::get_if<gguf::Array>(v) : nullptr;
}

} // namespace

std::expected<Tokenizer, std::string> Tokenizer::load(const gguf::File& meta)
{
    if (meta.get_string("tokenizer.ggml.model").value_or("") != "gpt2")
        return std::unexpected("tokenizer model is not byte-level BPE");
    const auto pre = meta.get_string("tokenizer.ggml.pre").value_or("");
    if (pre != "glm4" && pre != "glm5")
        return std::unexpected(std::format("unsupported pre-tokenizer '{}'", pre));

    const gguf::Array* tokens = array_of(meta, "tokenizer.ggml.tokens");
    const gguf::Array* types = array_of(meta, "tokenizer.ggml.token_type");
    const gguf::Array* merges = array_of(meta, "tokenizer.ggml.merges");
    if (!tokens || !types || !merges || types->items.size() != tokens->items.size())
        return std::unexpected("tokenizer vocabulary missing or inconsistent");

    Tokenizer t;
    t.ignore_merges_ = true;
    t.tokens_.reserve(tokens->items.size());
    for (size_t i = 0; i < tokens->items.size(); ++i) {
        const auto* s = std::get_if<std::string>(&tokens->items[i]);
        if (!s)
            return std::unexpected("tokenizer token is not a string");
        t.tokens_.push_back(*s);
        t.ids_.emplace(*s, int32_t(i));
        int64_t type = 1;
        if (auto* u = std::get_if<uint64_t>(&types->items[i]))
            type = int64_t(*u);
        else if (auto* s64 = std::get_if<int64_t>(&types->items[i]))
            type = *s64;
        t.types_.push_back(static_cast<TokenType>(type));
    }
    for (size_t i = 0; i < merges->items.size(); ++i) {
        const auto* s = std::get_if<std::string>(&merges->items[i]);
        if (!s)
            return std::unexpected("tokenizer merge is not a string");
        t.merge_rank_.emplace(*s, int32_t(i));
    }
    for (size_t i = 0; i < t.tokens_.size(); ++i) {
        const TokenType ty = t.types_[i];
        if (ty == TokenType::Control || ty == TokenType::UserDefined || ty == TokenType::Unknown)
            t.specials_.push_back(int32_t(i));
    }
    std::stable_sort(t.specials_.begin(), t.specials_.end(),
                     [&](int32_t a, int32_t b) { return t.tokens_[a].size() > t.tokens_[b].size(); });
    t.eos_ = int32_t(meta.get_int("tokenizer.ggml.eos_token_id").value_or(-1));
    t.eot_ = int32_t(meta.get_int("tokenizer.ggml.eot_token_id").value_or(-1));
    return t;
}

int32_t Tokenizer::find(std::string_view token_text) const
{
    auto it = ids_.find(std::string(token_text));
    return it == ids_.end() ? -1 : it->second;
}

std::vector<std::string> Tokenizer::split(std::string_view text) const
{
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    decode_utf8(text, cps, offs);
    std::vector<std::string> out;
    size_t at = 0;
    for (size_t len : split_lengths(cps)) {
        out.emplace_back(text.substr(offs[at], offs[at + len] - offs[at]));
        at += len;
    }
    return out;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const
{
    if (ignore_merges_) {
        if (auto it = ids_.find(word); it != ids_.end()) {
            out.push_back(it->second);
            return;
        }
    }
    // symbols are the UTF-8 characters of the byte-mapped word
    std::vector<std::string> sym;
    for (size_t i = 0; i < word.size();) {
        const auto b = static_cast<unsigned char>(word[i]);
        const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : 4;
        sym.push_back(word.substr(i, len));
        i += len;
    }
    while (sym.size() > 1) {
        int32_t best = std::numeric_limits<int32_t>::max();
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == std::numeric_limits<int32_t>::max())
            break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + std::ptrdiff_t(at) + 1);
    }
    for (const auto& s : sym) {
        if (auto it = ids_.find(s); it != ids_.end()) {
            out.push_back(it->second);
            continue;
        }
        // a symbol without a token: fall back to its characters
        for (size_t i = 0; i < s.size();) {
            const auto b = static_cast<unsigned char>(s[i]);
            const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : 4;
            if (auto it = ids_.find(s.substr(i, len)); it != ids_.end())
                out.push_back(it->second);
            i += len;
        }
    }
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t>& out) const
{
    const ByteMap& bm = byte_map();
    for (const auto& piece : split(text)) {
        std::string word;
        for (unsigned char b : piece)
            word += bm.to_text[b];
        bpe(word, out);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const
{
    // Fragments are either raw text or a special token id; longer special tokens split first.
    using Fragment = std::variant<std::string_view, int32_t>;
    std::vector<Fragment> frags{text};
    for (int32_t id : specials_) {
        if (!parse_special && types_[id] != TokenType::UserDefined)
            continue;
        const std::string& tok = tokens_[id];
        if (tok.empty())
            continue;
        std::vector<Fragment> next;
        for (const auto& f : frags) {
            const auto* raw = std::get_if<std::string_view>(&f);
            if (!raw) {
                next.push_back(f);
                continue;
            }
            std::string_view rest = *raw;
            for (size_t pos; (pos = rest.find(tok)) != std::string_view::npos;) {
                if (pos)
                    next.emplace_back(rest.substr(0, pos));
                next.emplace_back(id);
                rest.remove_prefix(pos + tok.size());
            }
            if (!rest.empty())
                next.emplace_back(rest);
        }
        frags = std::move(next);
    }

    std::vector<int32_t> out;
    for (const auto& f : frags) {
        if (const auto* id = std::get_if<int32_t>(&f))
            out.push_back(*id);
        else
            encode_plain(std::get<std::string_view>(f), out);
    }
    return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool special) const
{
    const ByteMap& bm = byte_map();
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= tokens_.size())
            continue;
        const TokenType ty = types_[id];
        if (ty == TokenType::Control || ty == TokenType::Unknown) {
            if (special)
                out += tokens_[id];
            continue;
        }
        if (ty == TokenType::UserDefined) {
            out += tokens_[id];
            continue;
        }
        std::vector<uint32_t> cps;
        std::vector<size_t> offs;
        decode_utf8(tokens_[id], cps, offs);
        for (uint32_t cp : cps) {
            auto it = bm.to_byte.find(cp);
            if (it != bm.to_byte.end())
                out += char(it->second);
            else
                append_utf8(out, cp);
        }
    }
    return out;
}

} // namespace mako::text
