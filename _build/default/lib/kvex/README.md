# kvex

[![Hex.pm](https://img.shields.io/hexpm/v/kvex.svg)](https://hex.pm/packages/kvex)
[![Hex Docs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/kvex)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Pure Erlang approximate k-nearest-neighbour vector search, powered by
[sied](https://hex.pm/packages/sied) SIMD NIFs (AVX2 / AVX-512 / NEON).

- **No training phase** — data-oblivious 1-bit quantization, vectors can be added at any time
- **Two-phase search** — SIMD POPCNT Hamming filter → SIMD dot-product rerank
- **Flat binary cache** — all vectors stored as a single ETS refc-binary; search never iterates over individual Erlang terms
- **Parallel reads** — ETS protected table; concurrent BEAM processes query the same index safely
- **Pure Erlang index layer** — only the SIMD kernels are in Rust (via sied)

## Performance

10 000 vectors × dim=128, K=10, OTP 27, AVX2:

| Operation | Latency |
|---|---|
| `add_batch/2` (10 000 vecs) | ~13 ms |
| `search/3` avg | ~60 µs |
| Throughput | **~16 000 queries/s** |

## Installation

```erlang
{deps, [{kvex, "0.2.1"}]}.
```

No Rust toolchain required — sied bundles a pre-compiled NIF.

## Quick start

```erlang
{ok, Ix} = kvex:new(128),

%% Insert a single vector
Vec = [rand:uniform() || _ <- lists:seq(1, 128)],
ok  = kvex:add(Ix, 42, Vec),

%% Batch insert (more efficient for large loads)
Pairs = [{I, [rand:uniform() || _ <- lists:seq(1, 128)]}
         || I <- lists:seq(1, 10000)],
ok = kvex:add_batch(Ix, Pairs),

%% Top-10 nearest neighbours
{ok, Results} = kvex:search(Ix, Vec, 10).
%% Results = [{Id, Score}]  sorted descending
```

Vectors can be passed as little-endian f32 binaries (`4 * Dim` bytes),
which avoids float-list traversal on hot paths:

```erlang
VBin = << <<X:32/float-little>> || X <- Vec >>,
ok   = kvex:add(Ix, 43, VBin),
{ok, Top} = kvex:search(Ix, VBin, 10).
```

## Cosine search

```erlang
%% L2-normalize before indexing and querying for cosine similarity:
{ok, NormVec} = sied:l2_normalize_f32(Vec),
ok = kvex:add(Ix, 1, NormVec),
{ok, Results} = kvex:cosine_search(Ix, QueryVec, 10).
```

## API

| Function | Description |
|---|---|
| `new(Dim)` / `new(Dim, Opts)` | Create an empty index |
| `add(Ix, Id, Vector)` | Insert one vector — O(N) flat cache rebuild |
| `add_batch(Ix, Pairs)` | Batch insert — O(batch) flat binary append |
| `search(Ix, Query, K)` | Top-K by dot product (two-phase) |
| `cosine_search(Ix, Query, K)` | Like search but auto-normalizes query |
| `normalize(Vec)` | L2-normalize a vector |
| `size(Ix)` | Number of indexed vectors |
| `delete(Ix)` | Free ETS table |

### Types

```erlang
-type id()     :: non_neg_integer() | binary().
-type vector() :: [float()] | binary().          %% binary = LE f32
-type opts()   :: #{bits => 2 | 3 | 4}.          %% bits option ignored in v0.2.0
```

### Scoring

`search/3` returns raw dot-product scores (higher = more similar). They are
comparable within a single index but not calibrated to cosine or L2. For
cosine semantics, L2-normalize your vectors before indexing and use
`cosine_search/3`.

## How it works

kvex maintains two structures in a single ETS table per index:

1. **Per-vector records** `{Id, F32Bin, BinVec}` — source of truth
2. **Flat cache** `{flat, F32FlatBin, BvecFlatBin, IdsTuple}` — all vectors
   concatenated into one refc-binary each

Search path (two NIF calls, no per-element Erlang term work):

```
sied:hamming_topk_flat/4   →  SIMD POPCNT on BvecFlat  →  top-C candidates
sied:dot_product_topk_flat/4  →  SIMD dot-product on F32Flat  →  top-K results
```

Default oversample factor: 10× (K=10 → 100 candidates for phase 1).

## Persistence

```erlang
%% Save index to disk
ok = kvex:dump(Ix, "/var/data/my_index.kvx"),

%% Restore (no re-quantization — loads flat binaries directly)
{ok, Ix2} = kvex:load("/var/data/my_index.kvx"),

%% Index is fully functional after load
ok = kvex:add(Ix2, NewId, NewVec),
{ok, Results} = kvex:search(Ix2, Query, 10).
```

File format: `KVEX` magic + version byte + dim + N + serialized IDs + raw f32 flat + raw bvec flat.
Restore is O(N) pointer setup (sub-binary slices into the flat buffer, no re-quantization).

## Building from source

```bash
rebar3 compile
rebar3 ct
```

## Links

- Hex.pm: [https://hex.pm/packages/kvex](https://hex.pm/packages/kvex)
- GitHub: [https://github.com/roquess/kvex](https://github.com/roquess/kvex)
- sied (SIMD NIFs): [https://hex.pm/packages/sied](https://hex.pm/packages/sied)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
