# Single Instruction Erlang Data (sied)

High-performance SIMD operations for Erlang through Rust NIFs.

[![Hex.pm](https://img.shields.io/hexpm/v/sied.svg)](https://hex.pm/packages/sied)
[![Hex Docs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/sied)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Motivation

Erlang excels at building concurrent, fault-tolerant systems, but numerical computations on large datasets can be a bottleneck. Modern CPUs offer SIMD (Single Instruction, Multiple Data) instructions that can process multiple data elements simultaneously, providing significant performance improvements for vectorized operations.

**sied** bridges this gap by exposing SIMD-accelerated mathematical operations to Erlang through a Rust NIF. The name combines **SI**MD with **E**rlang and **D**ata, representing the library's core purpose: bringing efficient data-parallel processing to the Erlang ecosystem.

## Features

- **Automatic SIMD optimization**: Compiler leverages AVX2, AVX-512, NEON, and other instruction sets via [simdeez](https://crates.io/crates/simdeez)
- **Flat-binary search primitives**: `hamming_topk_flat/4` and `dot_product_topk_flat/4` operate on concatenated binary buffers — zero per-element Erlang overhead, designed for ANN search
- **Simple API**: Consistent `{ok, Result} | {error, Reason}` interface throughout
- **Cross-platform**: Works across different CPU architectures with graceful scalar fallback

## Installation

Add to your `rebar.config`:

```erlang
{deps, [{sied, "0.2.0"}]}.
```

Or from GitHub:

```erlang
{deps, [
    {sied, {git, "https://github.com/roquess/sied.git", {branch, "main"}}}
]}.
```

## Building

```bash
rebar3 compile
```

**Requirements:**
- Erlang/OTP 24 or later
- Rust 1.70 or later (Cargo included)

The Rust toolchain must be available in your PATH. Visit [rustup.rs](https://rustup.rs) to install Rust.

For maximum performance (full AVX2/AVX-512), build with:

```bash
RUSTFLAGS="-C target-cpu=native" rebar3 as prod compile
```

## API Reference

All functions return `{ok, Result}` on success or `{error, Reason}` on failure.

### Basic Arithmetic

Element-wise operations on f32 or f64 vectors.

```erlang
{ok, Result} = sied:add_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).
%% Result = [5.0, 7.0, 9.0]

{ok, Result} = sied:multiply_f64([2.0, 3.0], [4.0, 5.0]).
%% Result = [8.0, 15.0]
```

Available: `add_f32/2`, `add_f64/2`, `subtract_f32/2`, `subtract_f64/2`,
`multiply_f32/2`, `multiply_f64/2`, `divide_f32/2`, `divide_f64/2`.

### Dot Product & Sum

```erlang
{ok, Dot} = sied:dot_product_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).
%% Dot = 32.0

{ok, Sum} = sied:sum_f32([1.0, 2.0, 3.0, 4.0, 5.0]).
%% Sum = 15.0
```

### Statistics

```erlang
{ok, Mean} = sied:mean_f32([1.0, 2.0, 3.0]).
{ok, Var}  = sied:variance_f32([1.0, 2.0, 3.0]).
{ok, Std}  = sied:std_dev_f32([1.0, 2.0, 3.0]).
```

Also available in f64 variants.

### Min / Max

```erlang
{ok, Min} = sied:min_f32([3.0, 1.0, 2.0]).   %% 1.0
{ok, Max} = sied:max_f32([3.0, 1.0, 2.0]).   %% 3.0
{ok, R}   = sied:min_elementwise_f32([1.0, 4.0], [2.0, 3.0]).  %% [1.0, 3.0]
```

### Unary Operations

```erlang
{ok, R} = sied:abs_f32([-1.0, 2.0, -3.0]).   %% [1.0, 2.0, 3.0]
{ok, R} = sied:sqrt_f32([4.0, 9.0, 16.0]).   %% [2.0, 3.0, 4.0]
{ok, R} = sied:negate_f32([1.0, -2.0]).       %% [-1.0, 2.0]
```

### L2 Norm and Normalization

```erlang
{ok, Norm} = sied:l2_norm_f32([3.0, 4.0]).       %% 5.0
{ok, Unit} = sied:l2_normalize_f32([3.0, 4.0]).   %% [0.6, 0.8]
{ok, Vecs} = sied:l2_normalize_batch_f32([[3.0, 4.0], [0.0, 2.0]]).
```

### Cosine Similarity

```erlang
{ok, Sim} = sied:cosine_similarity_f32([1.0, 0.0], [0.0, 1.0]).  %% 0.0
{ok, Sims} = sied:cosine_similarity_batch_f32(Query, [Vec1, Vec2, Vec3]).
```

### Batch Dot Product

One query against many vectors in a single NIF call.

```erlang
{ok, Scores} = sied:dot_product_batch_f32(Query, [Vec1, Vec2, Vec3]).

%% Binary variant — avoids float-list marshalling when vectors are stored
%% as little-endian f32 binaries (e.g. in ETS):
{ok, Scores} = sied:dot_product_batch_f32_bin(QBin, [Bin1, Bin2, Bin3]).
```

### Binary Quantization & Flat-Buffer ANN Search (v0.2.0)

These primitives are designed for two-phase approximate nearest-neighbour search on large indexes where the entire vector set is stored as a single concatenated binary (flat buffer) in ETS.

#### `to_binary_f32/1` and `to_binary_f32_bin/1`

1-bit quantize a vector: each dimension becomes 1 if above the mean, else 0. 128 dims → 16 bytes.

```erlang
{ok, BinVec} = sied:to_binary_f32([0.1, 0.9, 0.4, 0.8]).
%% BinVec = <<0b0101:4, ...>>  (packed bits)

%% Zero-copy variant when the vector is already a little-endian f32 binary:
{ok, BinVec} = sied:to_binary_f32_bin(F32Binary).
```

#### `hamming_topk_flat/4`

SIMD POPCNT over a flat binary buffer. Returns the indices of the `TopK` closest vectors, sorted ascending by Hamming distance. O(N) + O(K log K).

```erlang
%% BvecFlat = all binary-quantized vectors concatenated
%% VecLen   = byte size of one quantized vector
{ok, Indices} = sied:hamming_topk_flat(QBinVec, BvecFlat, VecLen, TopK).
%% Indices = [3, 17, 42, ...]  (0-based, sorted by distance)
```

Uses a max-heap of size K — O(K) memory regardless of corpus size.

#### `dot_product_topk_flat/4`

SIMD dot-product scoring of a candidate set selected from a flat f32 buffer. Designed to follow `hamming_topk_flat/4` in the two-phase search pipeline.

```erlang
%% F32Flat    = all f32 vectors concatenated
%% VecByteLen = dim * 4
%% Indices    = output of hamming_topk_flat/4
{ok, Scored} = sied:dot_product_topk_flat(QF32Bin, F32Flat, VecByteLen, Indices).
%% Scored = [{Score, Idx}, ...]  sorted by descending score
```

Uses zero-copy f32 reinterpretation — no heap allocation per candidate.

#### Two-phase ANN example

```erlang
%% Phase 1 — fast Hamming filter
CandCount = K * 10,
{ok, Cands} = sied:hamming_topk_flat(QBin, BvecFlat, VecByteLen, CandCount),

%% Phase 2 — precise dot-product rerank
{ok, Scored} = sied:dot_product_topk_flat(QF32Bin, F32Flat, VecF32ByteLen, Cands),
TopK = lists:sublist(Scored, K).
```

See [kvex](https://hex.pm/packages/kvex) for a complete k-NN index built on these primitives.

## Error Handling

Binary operations require vectors of equal length:

```erlang
case sied:add_f32([1.0, 2.0], [3.0]) of
    {ok, Result} -> io:format("Success: ~p~n", [Result]);
    {error, length_mismatch} -> io:format("vectors must have equal length~n")
end.
```

## Testing

```bash
rebar3 eunit
```

## Performance

The flat-buffer primitives (`hamming_topk_flat`, `dot_product_topk_flat`) are designed for high-throughput ANN search:

- `hamming_topk_flat` on 10 000 × 128-dim vectors: ~10 μs (release, AVX2)
- `dot_product_topk_flat` on 100 candidates × 128-dim: ~5 μs (release, AVX2)
- Combined two-phase search with [kvex](https://hex.pm/packages/kvex): **21 000+ queries/s** at 10 000 vectors, dim=128, K=10

General characteristics by vector size:

- **Small** (< 100 elements): NIF call overhead dominates — use pure Erlang for tiny vectors
- **Medium** (100–10 000 elements): Good speedup over pure Erlang
- **Large** (> 10 000 elements): Maximum SIMD benefit

## Project Structure

```
sied/
├── src/
│   ├── sied.app.src          # OTP application metadata
│   └── sied.erl              # Erlang API module
├── native/
│   └── sied/
│       ├── Cargo.toml        # Rust dependencies
│       └── src/
│           └── lib.rs        # Rust NIF implementation
├── test/
│   └── sied_tests.erl        # EUnit test suite
├── rebar.config
└── README.md
```

## Links

- GitHub: [https://github.com/roquess/sied](https://github.com/roquess/sied)
- Hex.pm: [https://hex.pm/packages/sied](https://hex.pm/packages/sied)
- kvex (ANN index using sied): [https://hex.pm/packages/kvex](https://hex.pm/packages/kvex)
- Rustler: [https://crates.io/crates/rustler](https://crates.io/crates/rustler)
- Simdeez: [https://crates.io/crates/simdeez](https://crates.io/crates/simdeez)

## License

Apache 2.0 — see [LICENSE](LICENSE).
