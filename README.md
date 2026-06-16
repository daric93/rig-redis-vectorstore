# rig-redis-vectorstore

[![crates.io](https://img.shields.io/crates/v/rig-redis-vectorstore.svg)](https://crates.io/crates/rig-redis-vectorstore)
[![docs.rs](https://img.shields.io/docsrs/rig-redis-vectorstore)](https://docs.rs/rig-redis-vectorstore)
[![CI](https://github.com/daric93/rig-redis-vectorstore/actions/workflows/ci.yaml/badge.svg)](https://github.com/daric93/rig-redis-vectorstore/actions/workflows/ci.yaml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Redis ([RediSearch](https://redis.io/docs/latest/develop/ai/search-and-query/)) vector store integration for the [Rig](https://crates.io/crates/rig-core) LLM framework.

`rig-redis-vectorstore` provides a `RedisVectorStore` that implements Rig's
`VectorStoreIndex`, `InsertDocuments`, and `VectorStoreIndexDyn` traits using
RediSearch vector similarity search (`FT.SEARCH` with KNN), with optional
metadata filtering.

> This integration was previously part of the `0xPlaygrounds/rig` monorepo and now
> lives here as an independent, community-maintained crate. It depends on `rig-core`
> from crates.io. It is published as `rig-redis-vectorstore` because the bare
> `rig-redis` name on crates.io is an unrelated placeholder.

## Features

- KNN vector similarity search via RediSearch (`top_n` and `top_n_ids`).
- Configurable distance metric: COSINE (default), L2, and inner product.
- Metadata filtering with a typed, injection-safe `Filter` builder (tag, numeric
  range, full-text, phrase, AND/OR/NOT), plus a `raw` escape hatch.
- Automatic extraction of document fields into filterable hash fields
  (`with_metadata_fields`).
- Index helpers: `create_index` and `validate_index` (checks metric + prefix).
- Document deletion by key (`delete`).
- Works over RESP2 and RESP3; embedding-model-agnostic; WASM-compatible bounds.

## Installation

```toml
[dependencies]
rig-core = "0.38"
rig-redis-vectorstore = "0.1"
```

You also need a Redis instance with the **RediSearch** module (e.g. Redis Stack,
or `redis/redis-stack-server`).

## Prerequisites: create the index

The store queries an existing RediSearch index; create it before use. The schema
must contain a `document` (TEXT) field, an `embedded_text` (TEXT) field, and a
VECTOR field (FLOAT32, COSINE distance). Add TAG/NUMERIC/TEXT fields for any
metadata you want to filter on, and a matching `PREFIX`:

```text
FT.CREATE products ON HASH PREFIX 1 products:
  SCHEMA document TEXT embedded_text TEXT
         embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
         category TAG price NUMERIC
```

You can also create it from code with `store.create_index(dims, &[("category", MetadataFieldType::Tag)])`,
and verify an existing index with `store.validate_index().await?` (checks
existence, COSINE distance, and prefix).

> Distance metric is configurable via `with_distance_metric` (COSINE default, plus
> L2 and inner product) and must match the index's `DISTANCE_METRIC`; scores are
> converted to "higher = more similar" per metric. It targets a single Redis node;
> both RESP2 and RESP3 replies are parsed. See the crate docs' "Limitations" section.

## Usage

```rust,ignore
use rig_redis_vectorstore::{Filter, RedisVectorStore};
use rig_core::vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest};

let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;

let store = RedisVectorStore::new(
    embedding_model,
    redis_client,
    "products".to_string(),
    "embedding".to_string(),
)
.await?
// Must match the index PREFIX so inserted docs are discoverable.
.with_key_prefix("products:".to_string())
// Extract these document fields into hash fields so they can be filtered.
.with_metadata_fields(vec!["category".to_string(), "price".to_string()]);

store.insert_documents(documents).await?;

let filter = Filter::eq("category", "Electronics")?.and(Filter::lt("price", 100.0)?);
let req = VectorSearchRequest::builder()
    .query("an affordable input device")
    .samples(5)
    .filter(filter)
    .build();

let results = store.top_n::<Product>(req).await?;
```

See [`examples/vector_search_redis.rs`](./examples/vector_search_redis.rs) for a
complete, runnable program (it uses a deterministic offline embedder, so it needs
no API key).

## Filtering

Build filters with the `Filter` type:

- `Filter::eq(field, value)` — numeric exact match (`@f:[v v]`), string/bool tag match (`@f:{v}`)
- `Filter::gt` / `lt` / `gte` / `lte` — numeric range; **non-numeric values return an error**
- `Filter::range(field, min, max)` / `range_exclusive`
- `Filter::tag_in(field, values)` — OR over tag values (empty list matches all)
- `Filter::text_contains` (token-AND) and `Filter::text_phrase` (exact phrase)
- `Filter::not`, `and`, `or`
- `Filter::raw(query)` — escape hatch; **do not pass unsanitized user input**

Field names and tag/text values are escaped automatically. The `escape_tag_value`
and `escape_text_value` helpers are exported for use with `Filter::raw`.

### Metadata filtering

Filters only match fields that exist in both the RediSearch index schema and the
stored hashes. Use `with_metadata_fields` to have top-level scalar document fields
(string/number/bool) extracted into hash fields during insertion. Reserved names
(`document`, `embedded_text`, and the vector field) are rejected; null/array/object
values are skipped with a warning.

## Testing

```bash
# Unit + filter tests (no services required)
cargo test

# Integration tests against a live RediSearch instance
podman run -d --name redis -p 6379:6379 redis/redis-stack-server:latest
REDIS_URL=redis://127.0.0.1:6379 cargo test --test integration_tests -- --ignored
```

Integration tests skip gracefully when neither `REDIS_URL` nor Docker/Podman is
available.

## Contributing

Contributions are welcome. Please **open an issue before sending a pull request**,
sign off your commits (`git commit -s`, per the DCO), and make sure the checks pass.
See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full guide (development setup,
running tests, and the PR checklist).

## License

Licensed under the [MIT License](./LICENSE). By contributing you agree that your
contributions are licensed under the same MIT terms (see
[CONTRIBUTING.md](./CONTRIBUTING.md)).
