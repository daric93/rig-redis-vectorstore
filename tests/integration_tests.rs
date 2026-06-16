//! Integration tests for the Redis vector store.
//!
//! These require a Redis instance with the RediSearch module (e.g. Redis Stack).
//! They are `#[ignore]` by default and run with:
//!
//! ```bash
//! cargo test --test integration_tests -- --ignored
//! ```
//!
//! Connection: set `REDIS_URL` to target an external instance, otherwise a
//! `redis/redis-stack` container is started via testcontainers. When neither
//! Docker nor a reachable RediSearch instance is available, the tests skip
//! gracefully (they print a notice and return) rather than failing.
#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_core::{
    Embed,
    embeddings::{
        EmbeddingsBuilder,
        embedding::{Embedding, EmbeddingError, EmbeddingModel},
    },
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_redis_vectorstore::{DistanceMetric, Filter, MetadataFieldType, RedisVectorStore};
use testcontainers::{
    ContainerAsync, GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};
use tokio::time::{Duration, sleep};

const REDIS_PORT: u16 = 6379;
const VECTOR_FIELD: &str = "embedding";
const DIMS: usize = 16;

/// Deterministic, offline embedding model: identical text always yields the
/// identical vector, so tests are reproducible without any network calls.
#[derive(Clone)]
struct StubModel;

fn stub_vec(text: &str) -> Vec<f64> {
    (0..DIMS)
        .map(|i| {
            // FNV-1a over the text bytes mixed with the dimension index.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in text.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h ^= i as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            (h % 1000) as f64 / 1000.0
        })
        .collect()
}

impl EmbeddingModel for StubModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        StubModel
    }

    fn ndims(&self) -> usize {
        DIMS
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        Ok(texts
            .into_iter()
            .map(|t| {
                let vec = stub_vec(&t);
                Embedding { document: t, vec }
            })
            .collect())
    }
}

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Product {
    name: String,
    category: String,
    price: f64,
    in_stock: bool,
    #[embed]
    description: String,
}

/// Per-test context holding the client, an optional owned container (kept alive
/// for the test's duration), and a randomized index name.
struct TestCtx {
    client: redis::Client,
    _container: Option<ContainerAsync<GenericImage>>,
    index: String,
}

/// Resolves a Redis endpoint, preferring `REDIS_URL`, else a testcontainer.
/// Returns `None` (with a printed notice) when nothing is available.
async fn connect() -> Option<(String, u16, Option<ContainerAsync<GenericImage>>)> {
    if let Ok(url) = std::env::var("REDIS_URL") {
        let trimmed = url.strip_prefix("redis://").unwrap_or(&url);
        let mut parts = trimmed.split(':');
        let host = parts.next().unwrap_or("127.0.0.1").to_string();
        let port = parts
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(REDIS_PORT);
        return Some((host, port, None));
    }

    let container = GenericImage::new("redis/redis-stack", "latest")
        .with_exposed_port(REDIS_PORT.tcp())
        .with_wait_for(WaitFor::Duration {
            length: Duration::from_secs(3),
        })
        .start()
        .await
        .ok()?;

    let port = container.get_host_port_ipv4(REDIS_PORT).await.ok()?;
    let host = container.get_host().await.ok()?.to_string();
    Some((host, port, Some(container)))
}

/// Probes for the RediSearch module via `FT._LIST`.
async fn verify_redisearch(client: &redis::Client) -> bool {
    let Ok(mut con) = client.get_multiplexed_async_connection().await else {
        return false;
    };
    redis::cmd("FT._LIST")
        .query_async::<redis::Value>(&mut con)
        .await
        .is_ok()
}

/// Prepares a skip-aware test context with a unique index name.
async fn prepare(base: &str) -> Option<TestCtx> {
    let Some((host, port, container)) = connect().await else {
        eprintln!("skipping {base}: Docker/Redis unavailable");
        return None;
    };
    let client = redis::Client::open(format!("redis://{host}:{port}")).ok()?;
    if !verify_redisearch(&client).await {
        eprintln!("skipping {base}: RediSearch module unavailable");
        return None;
    }
    let index = format!("{base}_{}", uuid::Uuid::new_v4().simple());
    Some(TestCtx {
        client,
        _container: container,
        index,
    })
}

/// Creates a HASH/VECTOR RediSearch index, optionally with metadata fields.
async fn create_index(
    client: &redis::Client,
    index: &str,
    with_metadata: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut con = client.get_multiplexed_async_connection().await?;

    let _: Result<String, _> = redis::cmd("FT.DROPINDEX")
        .arg(index)
        .arg("DD")
        .query_async(&mut con)
        .await;

    let prefix = format!("{index}:");
    let mut cmd = redis::cmd("FT.CREATE");
    cmd.arg(index)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(&prefix)
        .arg("SCHEMA")
        .arg("document")
        .arg("TEXT")
        .arg("embedded_text")
        .arg("TEXT")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(DIMS)
        .arg("DISTANCE_METRIC")
        .arg("COSINE");

    if with_metadata {
        cmd.arg("category")
            .arg("TAG")
            .arg("price")
            .arg("NUMERIC")
            .arg("in_stock")
            .arg("TAG");
    }

    cmd.query_async::<String>(&mut con).await?;
    sleep(Duration::from_millis(800)).await;
    Ok(())
}

/// Drops the index and its documents. Reported but non-fatal on failure.
async fn cleanup(client: &redis::Client, index: &str) {
    if let Ok(mut con) = client.get_multiplexed_async_connection().await {
        let res: Result<redis::Value, _> = redis::cmd("FT.DROPINDEX")
            .arg(index)
            .arg("DD")
            .query_async(&mut con)
            .await;
        if let Err(e) = res {
            eprintln!("cleanup of index {index} failed: {e}");
        }
    }
}

async fn store(ctx: &TestCtx, with_metadata: bool) -> RedisVectorStore<StubModel> {
    create_index(&ctx.client, &ctx.index, with_metadata)
        .await
        .unwrap();
    let mut s = RedisVectorStore::new(
        StubModel,
        ctx.client.clone(),
        ctx.index.clone(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", ctx.index));
    if with_metadata {
        s = s.with_metadata_fields(vec![
            "category".to_string(),
            "price".to_string(),
            "in_stock".to_string(),
        ]);
    }
    s
}

fn sample_words() -> Vec<Word> {
    vec![
        Word {
            id: "doc0".into(),
            definition: "a flurbo is a green alien that lives on cold planets".into(),
        },
        Word {
            id: "doc1".into(),
            definition: "a glarb-glarb is an ancient farming tool".into(),
        },
        Word {
            id: "doc2".into(),
            definition: "a linglingdong describes humans from the far side of the moon".into(),
        },
    ]
}

async fn insert_words(s: &RedisVectorStore<StubModel>, words: Vec<Word>) {
    let docs = EmbeddingsBuilder::new(StubModel)
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();
    s.insert_documents(docs).await.unwrap();
    sleep(Duration::from_millis(400)).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn vector_search_basic() {
    let Some(ctx) = prepare("it_basic").await else {
        return;
    };
    let s = store(&ctx, false).await;
    let words = sample_words();
    let target = words[2].definition.clone();
    insert_words(&s, words).await;

    // Query equals doc2's text, so it is the exact (distance ~0) nearest match.
    let req = VectorSearchRequest::builder()
        .query(target.clone())
        .samples(1)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2.definition, target);
    assert!(results[0].0.is_finite());

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn top_n_ids_returns_scored_ids() {
    let Some(ctx) = prepare("it_ids").await else {
        return;
    };
    let s = store(&ctx, false).await;
    insert_words(&s, sample_words()).await;

    let req = VectorSearchRequest::builder()
        .query("anything")
        .samples(2)
        .build();
    let results = s.top_n_ids(req).await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|(score, id)| score.is_finite() && !id.is_empty())
    );

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn threshold_filters_low_scores() {
    let Some(ctx) = prepare("it_threshold").await else {
        return;
    };
    let s = store(&ctx, false).await;
    insert_words(&s, sample_words()).await;

    let req = VectorSearchRequest::builder()
        .query("anything")
        .samples(10)
        .threshold(0.5)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();

    assert!(results.iter().all(|(score, _, _)| *score >= 0.5));

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn batch_insertion_writes_all_documents() {
    let Some(ctx) = prepare("it_batch").await else {
        return;
    };
    let s = store(&ctx, false).await;
    insert_words(&s, sample_words()).await;

    let mut con = ctx.client.get_multiplexed_async_connection().await.unwrap();
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg(format!("{}:*", ctx.index))
        .query_async(&mut con)
        .await
        .unwrap();
    assert!(
        keys.len() >= 3,
        "expected >= 3 inserted hashes, got {}",
        keys.len()
    );

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn empty_index_returns_no_results() {
    let Some(ctx) = prepare("it_empty").await else {
        return;
    };
    let s = store(&ctx, false).await;

    let req = VectorSearchRequest::builder()
        .query("query with no documents")
        .samples(5)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 0);

    cleanup(&ctx.client, &ctx.index).await;
}

fn sample_products() -> Vec<Product> {
    vec![
        Product {
            name: "Gaming Laptop".into(),
            category: "Electronics".into(),
            price: 1500.0,
            in_stock: true,
            description: "a high-end gaming laptop with discrete graphics".into(),
        },
        Product {
            name: "Wool Sweater".into(),
            category: "Clothing".into(),
            price: 75.0,
            in_stock: true,
            description: "a cozy wool sweater for winter".into(),
        },
        Product {
            name: "Mechanical Keyboard".into(),
            category: "Electronics".into(),
            price: 45.0,
            in_stock: false,
            description: "a budget-friendly mechanical keyboard".into(),
        },
    ]
}

async fn insert_products(s: &RedisVectorStore<StubModel>) {
    let docs = EmbeddingsBuilder::new(StubModel)
        .documents(sample_products())
        .unwrap()
        .build()
        .await
        .unwrap();
    s.insert_documents(docs).await.unwrap();
    sleep(Duration::from_millis(400)).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn metadata_filter_by_tag() {
    let Some(ctx) = prepare("it_meta_tag").await else {
        return;
    };
    let s = store(&ctx, true).await;
    insert_products(&s).await;

    let filter = Filter::eq("category", "Electronics").unwrap();
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();
    let results = s.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(_, _, p)| p.category == "Electronics"));

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn metadata_filter_by_numeric_range() {
    let Some(ctx) = prepare("it_meta_numeric").await else {
        return;
    };
    let s = store(&ctx, true).await;
    insert_products(&s).await;

    let filter = Filter::lt("price", 100.0).unwrap();
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();
    let results = s.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(_, _, p)| p.price < 100.0));

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn resp3_vector_search() {
    let Some((host, port, _container)) = connect().await else {
        eprintln!("skipping resp3_vector_search: Docker/Redis unavailable");
        return;
    };
    // Connect with the RESP3 protocol so the map-shaped FT.SEARCH reply is exercised.
    let client = redis::Client::open(format!("redis://{host}:{port}/?protocol=resp3")).unwrap();
    if !verify_redisearch(&client).await {
        eprintln!("skipping resp3_vector_search: RediSearch unavailable");
        return;
    }
    let index = format!("it_resp3_{}", uuid::Uuid::new_v4().simple());
    create_index(&client, &index, false).await.unwrap();

    let s = RedisVectorStore::new(
        StubModel,
        client.clone(),
        index.clone(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{index}:"));

    let words = sample_words();
    let target = words[2].definition.clone();
    insert_words(&s, words).await;

    let req = VectorSearchRequest::builder()
        .query(target.clone())
        .samples(1)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2.definition, target);

    cleanup(&client, &index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn validate_index_accepts_cosine_and_prefix() {
    let Some(ctx) = prepare("it_validate_ok").await else {
        return;
    };
    let s = store(&ctx, true).await;
    s.validate_index().await.unwrap();
    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn validate_index_rejects_non_cosine() {
    let Some(ctx) = prepare("it_validate_l2").await else {
        return;
    };
    let mut con = ctx.client.get_multiplexed_async_connection().await.unwrap();
    let _: Result<redis::Value, _> = redis::cmd("FT.DROPINDEX")
        .arg(&ctx.index)
        .arg("DD")
        .query_async(&mut con)
        .await;
    let _: String = redis::cmd("FT.CREATE")
        .arg(&ctx.index)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(format!("{}:", ctx.index))
        .arg("SCHEMA")
        .arg("document")
        .arg("TEXT")
        .arg("embedded_text")
        .arg("TEXT")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(DIMS)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query_async(&mut con)
        .await
        .unwrap();

    let s = RedisVectorStore::new(
        StubModel,
        ctx.client.clone(),
        ctx.index.clone(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", ctx.index));

    assert!(s.validate_index().await.is_err());
    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn create_index_helper_and_delete() {
    let Some(ctx) = prepare("it_create_delete").await else {
        return;
    };
    // Ensure a clean slate, then create the index via the helper.
    let mut con = ctx.client.get_multiplexed_async_connection().await.unwrap();
    let _: Result<redis::Value, _> = redis::cmd("FT.DROPINDEX")
        .arg(&ctx.index)
        .arg("DD")
        .query_async(&mut con)
        .await;

    let s = RedisVectorStore::new(
        StubModel,
        ctx.client.clone(),
        ctx.index.clone(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", ctx.index));

    s.create_index(DIMS, &[("category".to_string(), MetadataFieldType::Tag)])
        .await
        .unwrap();
    s.validate_index().await.unwrap();

    insert_words(&s, sample_words()).await;

    let ids = s
        .top_n_ids(
            VectorSearchRequest::builder()
                .query("anything")
                .samples(3)
                .build(),
        )
        .await
        .unwrap();
    let id_list: Vec<String> = ids.iter().map(|(_, id)| id.clone()).collect();
    let deleted = s.delete(&id_list).await.unwrap();
    assert_eq!(deleted as usize, id_list.len());

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn zero_samples_returns_empty_without_query() {
    let Some(ctx) = prepare("it_zero_samples").await else {
        return;
    };
    let s = store(&ctx, false).await;
    insert_words(&s, sample_words()).await;

    let req = VectorSearchRequest::builder()
        .query("anything")
        .samples(0)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();
    assert!(results.is_empty());

    cleanup(&ctx.client, &ctx.index).await;
}

/// Exercises a full create/insert/search cycle for a given distance metric.
///
/// `expect_target_first` is true for metrics where an identical vector is the
/// global nearest (COSINE, L2). For inner product on non-normalized vectors the
/// identical vector is not guaranteed to rank first, so we only assert presence
/// and monotonic ordering there.
async fn run_metric_cycle(metric: DistanceMetric, label: &str, expect_target_first: bool) {
    let Some(ctx) = prepare(label).await else {
        return;
    };
    let s = RedisVectorStore::new(
        StubModel,
        ctx.client.clone(),
        ctx.index.clone(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", ctx.index))
    .with_distance_metric(metric);

    s.create_index(DIMS, &[]).await.unwrap();
    s.validate_index().await.unwrap();

    let words = sample_words();
    let target = words[2].definition.clone();
    insert_words(&s, words).await;

    let req = VectorSearchRequest::builder()
        .query(target.clone())
        .samples(3)
        .build();
    let results = s.top_n::<Word>(req).await.unwrap();

    assert!(!results.is_empty(), "{label}: expected results");
    assert!(
        results.iter().all(|(score, _, _)| score.is_finite()),
        "{label}: all scores must be finite"
    );
    for w in results.windows(2) {
        assert!(
            w[0].0 >= w[1].0,
            "{label}: results must be ordered by descending score"
        );
    }
    if expect_target_first {
        assert_eq!(
            results[0].2.definition, target,
            "{label}: identical vector should rank first"
        );
    } else {
        assert!(
            results.iter().any(|(_, _, d)| d.definition == target),
            "{label}: target should be present in results"
        );
    }

    cleanup(&ctx.client, &ctx.index).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn distance_metric_cosine() {
    run_metric_cycle(DistanceMetric::Cosine, "it_metric_cosine", true).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn distance_metric_l2() {
    run_metric_cycle(DistanceMetric::L2, "it_metric_l2", true).await;
}

#[tokio::test]
#[ignore = "requires Docker/Podman or REDIS_URL with RediSearch"]
async fn distance_metric_inner_product() {
    run_metric_cycle(DistanceMetric::InnerProduct, "it_metric_ip", false).await;
}
