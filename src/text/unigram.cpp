#include "text/unigram.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace mako::text {

namespace {

std::string base64_decode(std::string_view s)
{
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        return c == '+' ? 62 : c == '/' ? 63 : -1;
    };
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        const int v = val(c);
        if (v < 0)
            continue;
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(char((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// The byte length of the UTF-8 character starting with lead byte c.
size_t char_len(unsigned char c)
{
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
}

uint32_t codepoint(std::string_view s, size_t i, size_t n)
{
    const auto* b = reinterpret_cast<const unsigned char*>(s.data() + i);
    if (n == 1)
        return b[0];
    uint32_t cp = b[0] & (0x7F >> n);
    for (size_t k = 1; k < n; ++k)
        cp = (cp << 6) | (b[k] & 0x3F);
    return cp;
}

// Marks that join the character before them (the common combining blocks, variation selectors, the joiner).
bool combining(uint32_t cp)
{
    return (cp >= 0x300 && cp <= 0x36F) || (cp >= 0x1AB0 && cp <= 0x1AFF) || (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE20 && cp <= 0xFE2F) || (cp >= 0xFE00 && cp <= 0xFE0F) ||
           cp == 0x3099 || cp == 0x309A || cp == 0x200D;
}

const std::string kSpace = "\xE2\x96\x81"; // ▁

} // namespace

std::expected<Unigram, std::string> Unigram::load_json(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::unexpected("cannot open " + path);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    glz::generic j;
    if (glz::read_json(j, text) || !j.is_object() || !j.contains("model"))
        return std::unexpected(path + ": not a tokenizer.json");
    auto& m = j["model"];
    if (!m.contains("type") || !m["type"].is_string() || m["type"].get_string() != "Unigram" || !m.contains("vocab"))
        return std::unexpected(path + ": not a unigram model");
    Unigram u;
    float lowest = std::numeric_limits<float>::max();
    for (auto& e : m["vocab"].get_array()) {
        const auto& pair = e.get_array();
        const std::string piece = pair[0].get_string();
        const float score = float(pair[1].get_number());
        u.ids_.emplace(piece, int32_t(u.pieces_.size()));
        u.longest_ = std::max(u.longest_, piece.size());
        u.pieces_.push_back(piece);
        u.scores_.push_back(score);
        lowest = std::min(lowest, score);
    }
    u.unk_score_ = lowest - 10.0f;
    if (m.contains("unk_id") && m["unk_id"].is_number())
        u.unk_ = int32_t(m["unk_id"].get_number());
    if (const auto it = u.ids_.find("</s>"); it != u.ids_.end())
        u.eos_ = it->second;
    // The normalizer: a precompiled map (sentencepiece's: the trie's size, the trie, its replacement strings).
    std::vector<const glz::generic*> norms;
    if (j.contains("normalizer") && j["normalizer"].is_object()) {
        auto& n = j["normalizer"];
        if (n.contains("normalizers"))
            for (auto& x : n["normalizers"].get_array())
                norms.push_back(&x);
        else
            norms.push_back(&n);
    }
    for (const auto* n : norms)
        if (n->contains("type") && (*n)["type"].get_string() == "Precompiled") {
            const std::string blob = base64_decode((*n)["precompiled_charsmap"].get_string());
            uint32_t size = 0;
            if (blob.size() < 4)
                return std::unexpected(path + ": bad precompiled map");
            std::memcpy(&size, blob.data(), 4);
            if (size % 4 || 4 + size > blob.size())
                return std::unexpected(path + ": bad precompiled map");
            u.trie_.resize(size / 4);
            std::memcpy(u.trie_.data(), blob.data() + 4, size);
            u.replacements_ = blob.substr(4 + size);
        }
    return u;
}

std::vector<uint32_t> Unigram::prefixes(std::string_view key) const
{
    std::vector<uint32_t> out;
    if (trie_.empty())
        return out;
    auto offset = [](uint32_t unit) { return (unit >> 10) << ((unit & (1u << 9)) >> 6); };
    size_t pos = offset(trie_[0]);
    for (unsigned char c : key) {
        if (c == 0)
            break;
        pos ^= c;
        if (pos >= trie_.size())
            break;
        const uint32_t unit = trie_[pos];
        if ((unit & ((1u << 31) | 0xFF)) != c)
            break;
        pos ^= offset(unit);
        if (pos >= trie_.size())
            break;
        if ((unit >> 8) & 1)
            out.push_back(trie_[pos] & ((1u << 31) - 1));
    }
    return out;
}

std::string Unigram::normalize(std::string_view text) const
{
    // The replacement of the first key that prefixes `s`, if any.
    auto transform = [&](std::string_view s, std::string& out) {
        const auto r = prefixes(s);
        if (r.empty() || r[0] >= replacements_.size())
            return false;
        out.append(replacements_.c_str() + r[0]);
        return true;
    };
    std::string out;
    for (size_t i = 0; i < text.size();) {
        size_t n = std::min(char_len(uint8_t(text[i])), text.size() - i), end = i + n;
        while (end < text.size()) {
            const size_t m = std::min(char_len(uint8_t(text[end])), text.size() - end);
            if (!combining(codepoint(text, end, m)))
                break;
            end += m;
        }
        if (!(end - i < 6 && transform(text.substr(i, end - i), out)))
            for (size_t k = i; k < end;) {
                const size_t m = std::min(char_len(uint8_t(text[k])), end - k);
                if (!transform(text.substr(k, m), out))
                    out.append(text.substr(k, m));
                k += m;
            }
        i = end;
    }
    return out;
}

void Unigram::segment(std::string_view s, std::vector<int32_t>& out) const
{
    // Viterbi over the pieces starting at each character: best[pos] is the best path ending at byte pos.
    struct Node {
        double score = -std::numeric_limits<double>::infinity();
        size_t prev = 0;
        int32_t id = -1;
        bool set = false;
    };
    std::vector<Node> best(s.size() + 1);
    best[0].score = 0;
    best[0].set = true;
    for (size_t pos = 0; pos < s.size();) {
        const size_t n = std::min(char_len(uint8_t(s[pos])), s.size() - pos);
        if (best[pos].set) {
            bool single = false;
            for (size_t len = 1; len <= longest_ && pos + len <= s.size(); ++len) {
                const auto it = ids_.find(std::string(s.substr(pos, len)));
                if (it == ids_.end())
                    continue;
                single |= len == n;
                const double sc = best[pos].score + scores_[size_t(it->second)];
                Node& e = best[pos + len];
                if (!e.set || sc > e.score)
                    e = {sc, pos, it->second, true};
            }
            if (!single) {
                const double sc = best[pos].score + unk_score_;
                Node& e = best[pos + n];
                if (!e.set || sc > e.score)
                    e = {sc, pos, unk_, true};
            }
        }
        pos += n;
    }
    std::vector<int32_t> rev;
    for (size_t pos = s.size(); pos > 0; pos = best[pos].prev)
        rev.push_back(best[pos].id);
    const size_t first = out.size();
    for (auto it = rev.rbegin(); it != rev.rend(); ++it)
        if (!(*it == unk_ && out.size() > first && out.back() == unk_))
            out.push_back(*it);
}

std::vector<int32_t> Unigram::encode(std::string_view text) const
{
    std::string t = normalize(text);
    // Runs of spaces made one, then spaces as ▁, one in front.
    std::string sp;
    for (char c : t)
        if (!(c == ' ' && !sp.empty() && sp.back() == ' '))
            sp.push_back(c);
    if (sp.empty())
        return {};
    std::string m;
    for (char c : sp)
        if (c == ' ')
            m += kSpace;
        else
            m.push_back(c);
    if (!m.starts_with(kSpace))
        m = kSpace + m;
    // Pieces each starting at a ▁.
    std::vector<int32_t> out;
    size_t start = 0;
    while (start < m.size()) {
        size_t next = m.find(kSpace, start + kSpace.size());
        if (next == std::string::npos)
            next = m.size();
        segment(std::string_view(m).substr(start, next - start), out);
        start = next;
    }
    return out;
}

} // namespace mako::text
