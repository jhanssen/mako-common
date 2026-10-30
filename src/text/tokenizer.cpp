#include "text/tokenizer.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <fstream>
#include <iterator>
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
bool is_punct(uint32_t cp) { return in_ranges(kPunctRanges, cp); }
bool is_symbol(uint32_t cp) { return in_ranges(kSymbolRanges, cp); }
bool is_mark(uint32_t cp) { return in_ranges(kMarkRanges, cp); }

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
//   's|'t|'re|'ve|'m|'ll|'d (any case) | [^\r\n\pL\pN]?\pL+ | \pN{1,max_digits} | ' '?[^\s\pL\pN]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<size_t> split_lengths(const std::vector<uint32_t>& c, uint32_t max_digits)
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
            while (j < i + max_digits && number(j))
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

// Lengths of the DeepSeek-V3 pre-tokenizer's pieces, in code points. Three patterns split in turn, each within the
// pieces the one before left (matches and the text between them alike):
//   \pN{1,3}
//   [\x{4E00}-\x{9FA5}\x{3040}-\x{30FF}]+
//   [ASCII punctuation][A-Za-z]+ | [^\r\n\pL\pP\pS]?[\pL\pM]+ | ' '?[\pP\pS]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<size_t> split_lengths_deepseek3(const std::vector<uint32_t>& c)
{
    using Piece = std::pair<size_t, size_t>; // [begin, end)
    // The pieces of each of `in` by `match` (the length of a match at i within [i, end), 0 for none): matches and the
    // runs between them, leftmost first.
    auto apply = [](const std::vector<Piece>& in, auto&& match) {
        std::vector<Piece> out;
        for (auto [b, e] : in) {
            size_t gap = b;
            for (size_t i = b; i < e;) {
                if (const size_t len = match(i, e)) {
                    if (gap < i)
                        out.emplace_back(gap, i);
                    out.emplace_back(i, i + len);
                    i += len;
                    gap = i;
                } else {
                    ++i;
                }
            }
            if (gap < e)
                out.emplace_back(gap, e);
        }
        return out;
    };
    std::vector<Piece> pieces{{0, c.size()}};
    pieces = apply(pieces, [&](size_t i, size_t e) {
        size_t j = i;
        while (j < e && j < i + 3 && is_number(c[j]))
            ++j;
        return j - i;
    });
    pieces = apply(pieces, [&](size_t i, size_t e) {
        auto cjk = [&](uint32_t cp) { return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF); };
        size_t j = i;
        while (j < e && cjk(c[j]))
            ++j;
        return j - i;
    });
    pieces = apply(pieces, [&](size_t i, size_t e) -> size_t {
        auto at = [&](size_t k) { return k < e; };
        auto ascii_letter = [&](size_t k) { return at(k) && c[k] < 0x80 && std::isalpha(int(c[k])); };
        auto ascii_punct = [&](size_t k) { return at(k) && c[k] < 0x80 && std::ispunct(int(c[k])); };
        auto lm = [&](size_t k) { return at(k) && (is_letter(c[k]) || is_mark(c[k])); };
        auto ps = [&](size_t k) { return at(k) && (is_punct(c[k]) || is_symbol(c[k])); };
        auto space = [&](size_t k) { return at(k) && is_space(c[k]); };
        if (ascii_punct(i) && ascii_letter(i + 1)) {
            size_t j = i + 1;
            while (ascii_letter(j))
                ++j;
            return j - i;
        }
        {
            size_t j = i;
            if (!is_newline(c[i]) && !is_letter(c[i]) && !is_punct(c[i]) && !is_symbol(c[i]) && lm(i + 1))
                ++j;
            if (lm(j)) {
                while (lm(j))
                    ++j;
                return j - i;
            }
        }
        {
            size_t j = i;
            if (c[j] == ' ' && ps(j + 1))
                ++j;
            if (ps(j)) {
                while (ps(j))
                    ++j;
                while (at(j) && is_newline(c[j]))
                    ++j;
                return j - i;
            }
        }
        if (space(i)) {
            size_t end = i;
            while (space(end))
                ++end;
            size_t last_nl = e;
            for (size_t j = i; j < end; ++j)
                if (is_newline(c[j]))
                    last_nl = j;
            if (last_nl != e) {
                size_t j = last_nl + 1;
                while (at(j) && is_newline(c[j]))
                    ++j;
                return j - i;
            }
            if (end == e)
                return end - i;
            return end - i >= 2 ? end - i - 1 : 1;
        }
        return 0;
    });
    std::vector<size_t> out;
    for (auto [b, e] : pieces)
        out.push_back(e - b);
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
    const bool deepseek = pre == "joyai-llm" || pre == "deepseek-v3";
    if (pre != "glm4" && pre != "glm5" && !deepseek)
        return std::unexpected(std::format("unsupported pre-tokenizer '{}'", pre));

    const gguf::Array* tokens = array_of(meta, "tokenizer.ggml.tokens");
    const gguf::Array* types = array_of(meta, "tokenizer.ggml.token_type");
    const gguf::Array* merges = array_of(meta, "tokenizer.ggml.merges");
    if (!tokens || !types || !merges || types->items.size() != tokens->items.size())
        return std::unexpected("tokenizer vocabulary missing or inconsistent");

    Tokenizer t;
    t.deepseek_ = deepseek;
    t.ignore_merges_ = !deepseek; // the vocabulary's whole-word entries win over the merges only for GLM's
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

// model.vocab (text -> id) and model.merges ([left, right] pairs, by rank), added_tokens (special ones are control
// tokens, the rest user-defined); Qwen's split (GLM's with single digits), merges applied as they are.
std::expected<Tokenizer, std::string> Tokenizer::load_json(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected("cannot open " + path);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (glz::read_json(j, text) || !j.is_object() || !j.contains("model") || !j["model"].is_object())
        return std::unexpected(path + ": not a tokenizer.json");
    const auto& model = j["model"];
    if (!model.contains("type") || model["type"].get_string() != "BPE" || !model.contains("vocab") ||
        !model.contains("merges"))
        return std::unexpected(path + ": not a BPE tokenizer");
    Tokenizer t;
    t.max_digits_ = 1;
    std::vector<std::pair<int32_t, std::string>> entries;
    for (const auto& [tok, id] : model["vocab"].get_object())
        entries.emplace_back(int32_t(id.get_number()), tok);
    std::vector<std::pair<int32_t, bool>> added;
    if (j.contains("added_tokens"))
        for (const auto& a : j["added_tokens"].get_array()) {
            entries.emplace_back(int32_t(a["id"].get_number()), a["content"].get_string());
            added.emplace_back(int32_t(a["id"].get_number()), a.contains("special") && a["special"].get_boolean());
        }
    int32_t n = 0;
    for (const auto& [id, tok] : entries)
        n = std::max(n, id + 1);
    t.tokens_.assign(size_t(n), {});
    t.types_.assign(size_t(n), TokenType::Unused);
    for (const auto& [id, tok] : entries) {
        if (id < 0)
            return std::unexpected(path + ": negative token id");
        t.tokens_[size_t(id)] = tok;
        t.types_[size_t(id)] = TokenType::Normal;
        t.ids_.emplace(tok, id);
    }
    for (const auto& [id, special] : added)
        t.types_[size_t(id)] = special ? TokenType::Control : TokenType::UserDefined;
    const auto& merges = model["merges"].get_array();
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto& m = merges[i];
        const std::string pair = m.is_array() ? m[0].get_string() + " " + m[1].get_string() : m.get_string();
        t.merge_rank_.emplace(pair, int32_t(i));
    }
    for (size_t i = 0; i < t.tokens_.size(); ++i) {
        const TokenType ty = t.types_[i];
        if (ty == TokenType::Control || ty == TokenType::UserDefined)
            t.specials_.push_back(int32_t(i));
    }
    std::stable_sort(t.specials_.begin(), t.specials_.end(),
                     [&](int32_t a, int32_t b) { return t.tokens_[a].size() > t.tokens_[b].size(); });
    t.eos_ = t.find("<|endoftext|>");
    t.eot_ = t.find("<|im_end|>");
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
    for (size_t len : deepseek_ ? split_lengths_deepseek3(cps) : split_lengths(cps, max_digits_)) {
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
    const std::pair<size_t, size_t> all{0, text.size()};
    return encode(text, parse_special ? std::span<const std::pair<size_t, size_t>>{} : std::span(&all, 1));
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, std::span<const std::pair<size_t, size_t>> literal) const
{
    const auto in_literal = [&](size_t begin, size_t end) {
        // The first range ending after begin is the only one that can overlap [begin, end).
        auto it = std::ranges::upper_bound(literal, begin, {}, &std::pair<size_t, size_t>::second);
        return it != literal.end() && it->first < end;
    };
    // Fragments are either raw text or a special token id; longer special tokens split first.
    using Fragment = std::variant<std::string_view, int32_t>;
    std::vector<Fragment> frags{text};
    for (int32_t id : specials_) {
        const bool control = types_[id] != TokenType::UserDefined;
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
            for (size_t pos, from = 0; (pos = rest.find(tok, from)) != std::string_view::npos;) {
                const size_t at = size_t(rest.data() - text.data()) + pos;
                if (control && in_literal(at, at + tok.size())) {
                    from = pos + 1;
                    continue;
                }
                if (pos)
                    next.emplace_back(rest.substr(0, pos));
                next.emplace_back(id);
                rest.remove_prefix(pos + tok.size());
                from = 0;
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
