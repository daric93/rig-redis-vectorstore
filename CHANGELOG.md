# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-16

### Fixed
- Documentation only: README badges. Replaced the unreliable shields.io
  `crates/l` license badge (which renders "invalid" even for a valid MIT license)
  with a static MIT badge, and refreshed the crates.io version badge URL. Published
  as a patch so the crates.io-rendered README (a per-version snapshot) shows the
  corrected badges. No code or API changes.

## [0.1.0] - 2026-06-16

Initial standalone release of the Redis (RediSearch) vector store integration for
the Rig framework, extracted from the `0xPlaygrounds/rig` monorepo and rebased onto
the current `rig-core` filter API.

### Added
- `RedisVectorStore` implementing `VectorStoreIndex`, `InsertDocuments`, and
  `VectorStoreIndexDyn` over RediSearch `FT.SEARCH` KNN queries (COSINE distance).
- `Filter` type implementing `SearchFilter` (`Value = RedisNumber`, finite-checked),
  with fallible numeric range constructors that reject non-numeric values, plus
  `tag_in`, `text_contains`, `text_phrase`, `range`, `range_exclusive`, `not`,
  `and`, `or`, and a `raw` escape hatch.
- Automatic escaping of field names and tag/text values; exported
  `escape_tag_value` and `escape_text_value` helpers.
- `with_metadata_fields` for extracting top-level scalar document fields into Redis
  hash fields during insertion, enabling metadata filtering, with a reserved-field
  guard and skip-with-warning behavior for unsupported values.
- `validate_index` to verify (via `FT.INFO`) that the index exists, uses the
  configured distance metric, and is configured with the store's key prefix.
- `with_distance_metric` and `DistanceMetric` (Cosine/L2/InnerProduct), with
  per-metric distance→similarity conversion (`with_distance_metric` must match the
  index `DISTANCE_METRIC`).
- `create_index` convenience helper and `MetadataFieldType` (Tag/Numeric/Text).
- `delete` to remove documents by key (`UNLINK`).
- RESP2 and RESP3 `FT.SEARCH` reply parsing.
- Guards for `samples == 0` (returns empty) and non-finite query embeddings.
- Graceful response parsing that skips empty/unparseable documents and warns on a
  missing `__vector_score` rather than aborting the whole result set.
- Unit tests (filter syntax, metadata behavior, RESP2/RESP3 parsing, distance-metric
  score conversion) and gated integration tests (testcontainers / `REDIS_URL`,
  RediSearch probe, randomized index names, RESP3, index validation, create/delete,
  and per-metric COSINE/L2/inner-product cycles).
- Runnable, credential-free example (`examples/vector_search_redis.rs`).
