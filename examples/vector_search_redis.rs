//! Vector search over Redis (RediSearch) with metadata filtering.
//!
//! This example is fully self-contained: it uses a small **deterministic, offline
//! embedding model** so it runs without any API credentials, and it creates the
//! RediSearch index itself. In a real application you would replace
//! `DemoEmbeddingModel` with a provider model such as
//! `openai::Client::from_env().embedding_model(openai::TEXT_EMBEDDING_ADA_002)`.
//!
//! Prerequisites: a Redis instance with the RediSearch module (e.g. Redis Stack)
//! reachable at `REDIS_URL` (defaults to `redis://127.0.0.1:6379`).
//!
//! Run with: `cargo run --example vector_search_redis --features rig-core/derive`

use rig_core::{
    Embed,
    embeddings::{
        EmbeddingsBuilder,
        embedding::{Embedding, EmbeddingError, EmbeddingModel},
    },
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_redis_vectorstore::{Filter, RedisVectorStore};

const DIMS: usize = 32;
const VECTOR_FIELD: &str = "embedding";
const INDEX: &str = "products";

/// Deterministic offline embedding model (demo only — no network).
#[derive(Clone)]
struct DemoEmbeddingModel;

fn demo_vec(text: &str) -> Vec<f64> {
    (0..DIMS)
        .map(|i| {
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

impl EmbeddingModel for DemoEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        DemoEmbeddingModel
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
                let vec = demo_vec(&t);
                Embedding { document: t, vec }
            })
            .collect())
    }
}

#[derive(Embed, Clone, serde::Serialize, serde::Deserialize, Debug)]
struct Product {
    name: String,
    category: String,
    price: f64,
    #[embed]
    description: String,
}

async fn create_index(client: &redis::Client) -> anyhow::Result<()> {
    let mut con = client.get_multiplexed_async_connection().await?;
    // Recreate the index for a clean demo run.
    let _: Result<redis::Value, _> = redis::cmd("FT.DROPINDEX")
        .arg(INDEX)
        .arg("DD")
        .query_async(&mut con)
        .await;
    let _: String = redis::cmd("FT.CREATE")
        .arg(INDEX)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(format!("{INDEX}:"))
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
        .arg("COSINE")
        .arg("category")
        .arg("TAG")
        .arg("price")
        .arg("NUMERIC")
        .query_async(&mut con)
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url)?;

    create_index(&redis_client).await?;

    let store = RedisVectorStore::new(
        DemoEmbeddingModel,
        redis_client,
        INDEX.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await?
    .with_key_prefix(format!("{INDEX}:"))
    .with_metadata_fields(vec!["category".to_string(), "price".to_string()]);

    let products = vec![
        Product {
            name: "Gaming Laptop".to_string(),
            category: "Electronics".to_string(),
            price: 1500.0,
            description: "A high-end gaming laptop with discrete graphics".to_string(),
        },
        Product {
            name: "Mechanical Keyboard".to_string(),
            category: "Electronics".to_string(),
            price: 45.0,
            description: "A budget-friendly mechanical keyboard".to_string(),
        },
        Product {
            name: "Wool Sweater".to_string(),
            category: "Clothing".to_string(),
            price: 75.0,
            description: "A cozy wool sweater for winter".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(DemoEmbeddingModel)
        .documents(products)?
        .build()
        .await?;
    store.insert_documents(documents).await?;

    // Give the index a moment to ingest.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Vector search restricted to Electronics priced under 100.
    let filter = Filter::eq("category", "Electronics")?.and(Filter::lt("price", 100.0)?);
    let req = VectorSearchRequest::builder()
        .query("an affordable input device")
        .samples(5)
        .filter(filter)
        .build();

    println!("Electronics under $100:");
    let results = store.top_n::<Product>(req).await?;
    for (score, id, product) in results {
        println!(
            "  {score:.4}  {id}  {}  (${:.2})",
            product.name, product.price
        );
    }

    Ok(())
}
