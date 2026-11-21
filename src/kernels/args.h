// The common kernels' argument structs, one definition for the kernels and the code dispatching them (C++:
// mako::kernels; Metal: the global namespace), so the two cannot disagree on a layout. Fields a caller leaves out
// of an aggregate initializer are zero.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define MAKO_GPU_PTR(type) device type*
#else
#include <cstdint>
#define MAKO_GPU_PTR(type) uint64_t // a GPU address
namespace mako::kernels {
#endif

// gemm.metal

struct GemmArgs {
    uint32_t t, k, n, row_bytes;
    uint32_t ldy; // y row stride in elements
    uint32_t ldx; // x row stride in elements
    // gemm only: batch > 1 matrices, grid y t_tiles a matrix; matrix m's weights w_mat bytes, x x_mat and y y_mat
    // elements after matrix 0's (batch 0 or 1: one)
    uint32_t batch = 0, t_tiles = 0, w_mat = 0, x_mat = 0, y_mat = 0;
};

struct QuantArgs {
    uint32_t cols, ldx, rows;
};

struct HalfArgs {
    uint32_t cols, ldx;
};

struct PvArgs {
    uint32_t heads, dim, n_kv;
    uint32_t keys_per_split; // a multiple of 64
};

struct GatherArgs {
    uint32_t heads;  // query rows per token
    uint32_t dim;    // latent width (multiple of 256)
    uint32_t stride; // list and score row length
    uint32_t splits; // attn_gather_pv: key ranges per token
    uint32_t tokens; // attn_gather_pv: partial is [splits][tokens][heads][dim]
};

struct IdxDotArgs {
    uint32_t heads, dim, pools; // pools: of the last token
    uint32_t stride;            // dots row length
};

// gemv.metal

struct ShortArgs {
    uint32_t k, n, row_bytes;
    uint32_t tokens; // 1..8: x rows of k, y rows of n
};

struct ShortBatchArgs {
    uint32_t k, n, row_bytes;
    uint32_t tokens;        // 1..8
    uint64_t w_mat;         // bytes between matrices
    uint32_t x_tok, x_mat;  // floats between tokens' / matrices' x rows
    uint32_t y_tok, y_mat;  // floats between tokens' / matrices' y rows
};

// gemv_slots.metal

struct GemvJob {
    MAKO_GPU_PTR(const char) w;      // weights: matrix 0 / slot 0
    MAKO_GPU_PTR(const char) x;      // activations: token 0, matrix 0
    MAKO_GPU_PTR(char) y;            // outputs: token 0, matrix 0
    MAKO_GPU_PTR(const int) ids;     // indexed: the slot ids (a row per token); null otherwise
    MAKO_GPU_PTR(const char) w_tail; // indexed: the slots from `split` on
    uint64_t nb02;                   // bytes between matrices or slots
    uint64_t nb01;                   // bytes between rows
    uint64_t x_tok, x_mat;           // bytes between tokens' / matrices' x rows (x_mat 0: shared)
    uint64_t y_tok, y_mat;           // bytes between tokens' / matrices' y rows
    uint32_t ids_stride;             // elements between tokens' id rows
    uint32_t split;
    uint32_t type;                   // JobType
    uint32_t k, n, nr0;
    uint32_t tg0, tgs_x;             // first threadgroup; threadgroups per matrix
    uint32_t nb00;                   // block bytes
};
enum JobType : uint32_t {
    JOB_Q6_K = 0, JOB_Q8_0 = 1, JOB_F32_4 = 2, JOB_IQ2_XXS = 3, JOB_IQ3_XXS = 4, JOB_IQ2_S = 5, JOB_IQ3_S = 6,
    JOB_Q5_K = 7, JOB_IQ2_XS = 8, JOB_IQ1_S = 9, JOB_IQ1_M = 10, JOB_IQ4_XS = 11,
};

// moe_route.h

struct RouteArgs {
    uint32_t n_expert, n_used, norm;
    float scale;
    uint32_t lookup;    // 1: look the selection up in slots (one token); 2: also count it in `counts`
    uint32_t score;     // 0: sigmoid, 1: sqrt(softplus)
    float miss_penalty = 0.0f; // > 0: subtracted from the selection score (not the weight) of an expert slots lacks
};

// rows.h

struct NormArgs {
    uint32_t n;       // row length
    uint32_t wn;      // weight length (the weight repeats every wn elements); 0: no weight
    float eps;
    uint32_t ystride = 0; // rms_norm: y row stride in elements (0: n)
};

// ops.metal

struct HeadsArgs {
    uint32_t rows;  // tokens
    uint32_t ld;    // x row stride in elements
    uint32_t col0;  // the first head's column
    uint32_t dim;   // per head
    uint32_t n_pad; // rows of a head in y
    uint32_t row0;  // y row of token 0
    uint32_t rope;  // 0 none; 1 halves (i with i + dim/2); 2 adjacent pairs (2i with 2i + 1)
};

struct AttnArgs {
    uint32_t n_q;    // queries
    uint32_t q_pad;  // rows of a query head (a multiple of 32)
    uint32_t n_k;    // keys
    uint32_t k_pad;  // rows of a key / value head (a multiple of 128)
    uint32_t heads, kv_heads;
    uint32_t q_pos0; // the position of query 0 among the keys
    uint32_t block;  // positions from here on see each other both ways (the image); below, causal
    uint32_t width;  // o row length
    float scale;
    uint32_t limits = 0; // 1: query position p sees the keys before limits[p] instead (buffer 5, non-decreasing)
};

struct HeadRowsArgs {
    uint32_t dim, src_pad, dst_pad, rows;
};

struct HeadsNormArgs {
    uint32_t ld, col0, dim, n_pad, row0, rope; // as HeadsArgs
    float eps;
};

struct ScanArgs {
    uint32_t head_dim;   // 128
    uint32_t tokens;
    uint32_t qkv_stride; // q, k, v elements between tokens
    uint32_t inner;      // g and o elements between tokens
    uint32_t heads;      // beta elements between tokens
    uint32_t snap;       // nonzero: also write the state after this many tokens to snap_state (for a rollback)
};

struct SwigluArgs {
    uint32_t n;
    float limit; // <= 0: no clamp
};

struct HeadsPartialArgs {
    uint32_t ld, col0; // x rows ld apart; head h at col0 + h * stride
    uint32_t stride;
    uint32_t dim;      // per head (at most 256)
    uint32_t rot;      // the first rot values rotate, i with i + rot / 2 (0: none)
    uint32_t n_pad;    // rows of a head in y
    float eps;         // 0: no norm
    uint32_t mapped = 0; // row t of y comes from x row map[t] (and its angles), a zero row where map[t] < 0
};

struct LoraArgs {
    uint32_t k, rows, rank; // the matrix's row length, the rows adapted, the adapter's rank
    uint32_t b_rot;         // row r of the matrix takes B's row (r + b_rot) % rows
    float scale;
};

struct LnAffineArgs {
    uint32_t n;
    float eps;
};

struct LayerNormArgs {
    uint32_t n;
    float eps;
    uint32_t ystride = 0; // y row stride in elements (0: n)
};

struct Im2colArgs {
    uint32_t h, w, c;  // input image
    uint32_t p0, rows; // output pixels [p0, p0 + rows)
    uint32_t kpad;     // columns, 9 * c padded to a multiple of 32
    uint32_t mode;     // the padding's: 0 zeros, 1 reflect
};

struct Im2col3dArgs {
    int32_t h, w, c, p0, rows, kpad, mode, ow, stride, pad, f0, T;
};

struct Im2colS2Args {
    uint32_t h, w, c, p0, rows, kpad, ow;
    uint32_t pad;  // above and left of the image: 1, or 0 (then only below and right)
    uint32_t mode; // the padding's: 0 zeros, 1 reflect
};

struct GnArgs {
    uint32_t rows, c, groups, chunk; // chunk: rows a threadgroup of gn_partial sums
    float eps;
    uint32_t silu;
};

struct CubeArgs {
    // Cubes of 64 rows; rows of a head in the tiles pooled (q's and k's can differ); the width.
    uint32_t n_cubes, k_pad, dim;
    uint32_t topk;   // cube_topk: cubes kept per query cube
    uint32_t local;  // cube_topk: 1 keeps the query cube itself first, 3 also the cubes `stride` before and after
    uint32_t stride; // cubes a frame group apart (the temporal neighbours)
};

struct SparseAttnArgs {
    uint32_t n_q, q_pad, n_k, k_pad, heads, kv_heads, width; // as AttnArgs (rows in the tiles' order)
    float scale;
    uint32_t n_cubes, max_sel; // the first n_cubes query tiles are cubes; sel rows hold max_sel entries
    uint32_t dense0, dense1;   // the dense rows [dense0, dense1), attended by every cube's queries
};

struct RowModArgs {
    uint32_t n;         // row width
    float eps;
    uint32_t shift_off; // of the shift within a row's modulation, in floats
    uint32_t scale_off; // of the scale
};

struct GateRowsArgs {
    uint32_t n;
    uint32_t gate_off; // of the gate within a row's modulation, in floats
};

// ops.metal: Qwen3.5's text layers

struct GdnConvArgs {
    uint32_t channels;
    uint32_t taps; // the last one multiplies the current token (at most 8)
    uint32_t tokens;
};

struct GdnPrepArgs {
    uint32_t k_heads, v_heads; // of 128: each key head serves v_heads / k_heads value heads
    uint32_t conv_width;       // a conv row: q (k_heads), k (k_heads), v (v_heads)
    float eps;                 // of the l2 norm: x / sqrt(sum x^2 + eps)
};

struct GatedRmsArgs {
    uint32_t heads; // of 128
    float eps;
};

struct GateHeadsArgs {
    uint32_t heads, dim;
    uint32_t ld, col0; // the gate of head h at src[t][col0 + h * stride]
    uint32_t stride;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(GemvJob) == 128);
static_assert(sizeof(ShortBatchArgs) == 40);
} // namespace mako::kernels
#endif
#undef MAKO_GPU_PTR
