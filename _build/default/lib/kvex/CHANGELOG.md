# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-20

### Added
- Initial release.
- `kvex:new/1`, `kvex:new/2` — create a typed, dimension-fixed index.
- `kvex:add/3`, `kvex:add_batch/2` — single and atomic-batch inserts.
- `kvex:search/3` — top-K similarity search (dirty CPU scheduler).
- `kvex:size/1`, `kvex:delete/1` — introspection and explicit release.
- Vectors accepted as either `[float()]` lists or little-endian f32
  binaries.
- Parallel reads via `RwLock` — concurrent searches share a read lock.
- NIF backed by the `turbovec` crate (TurboQuant, Google Research
  ICLR 2026).
