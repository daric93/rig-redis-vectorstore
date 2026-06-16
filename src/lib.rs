//! Redis vector store integration for Rig.
//!
//! Provides a [`RedisVectorStore`] that implements Rig's [`VectorStoreIndex`] and
//! [`InsertDocuments`] traits using RediSearch's vector similarity search (`FT.SEARCH`).
//!
//! # Prerequisites
//!
//! The RediSearch index must be created before using this store. The expected schema is:
//! - A HASH-based index with the specified prefix
//! - A `document` field of type TEXT (stores serialized JSON)
//! - An `embedded_text` field of type TEXT (stores the source text)
//! - A vector field (configurable name) of type VECTOR with FLOAT32 elements
//! - Optionally, additional fields for metadata filtering (TAG, NUMERIC, etc.)
//!
//! # Distance Metric
//!
//! The metric is configurable via [`RedisVectorStore::with_distance_metric`] and must
//! match the index's `DISTANCE_METRIC`. [`DistanceMetric::Cosine`] (the default),
//! [`DistanceMetric::L2`], and [`DistanceMetric::InnerProduct`] are supported. Returned
//! distances are converted to similarity scores (higher = more similar) per metric; see
//! [`DistanceMetric`]. Use [`RedisVectorStore::validate_index`] to confirm the index
//! agrees with the configured metric.
//!
//! # Metadata Filtering
//!
//! To enable filtering on document fields during search, configure metadata fields
//! via [`RedisVectorStore::with_metadata_fields`]. These fields are extracted from
//! the serialized document JSON during insertion and written as separate hash fields,
//! making them available for RediSearch filter queries. Your index schema must declare
//! these fields with appropriate types (TAG, NUMERIC, TEXT) for filters to work.
//!
//! # Limitations
//!
//! - **Single-node only.** Inserts are pipelined across multiple keys, which is not
//!   compatible with Redis Cluster (CROSSSLOT). Cluster support is a planned follow-up.
//! - **Key prefix must match the index `PREFIX`**, otherwise inserted documents are
//!   stored but never indexed.
//! - **Multiple embeddings per document** produce multiple independently searchable
//!   hashes, so a single logical document may appear more than once in results.
//!
//! Both RESP2 and RESP3 `FT.SEARCH` reply shapes are parsed.
//!
//! # Example
//! ```ignore
//! use rig_redis_vectorstore::RedisVectorStore;
//!
//! let store = RedisVectorStore::new(
//!     embedding_model,
//!     redis_client,
//!     "my_index".into(),
//!     "embedding".into(),
//! )
//! .await?
//! .with_key_prefix("doc:".to_string())
//! .with_metadata_fields(vec!["category".to_string(), "price".to_string()]);
//! ```

pub mod filter;

pub use filter::Filter;
use redis::aio::ConnectionManager;
use rig_core::{
    Embed, OneOrMany,
    embeddings::embedding::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, TopNResults, VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn,
        request::{Filter as CoreFilter, VectorSearchRequest},
    },
    wasm_compat::WasmBoxedFuture,
};
use serde::{Deserialize, Serialize};

/// Redis vector store implementation using RediSearch vector similarity search.
///
/// Uses Redis's `FT.SEARCH` command with KNN vector queries for similarity search.
/// Internally holds a [`ConnectionManager`] for automatic reconnection on transient failures.
///
/// # Key Prefix
///
/// If your RediSearch index uses a `PREFIX` configuration (e.g., `PREFIX 1 doc:`),
/// you **must** call [`RedisVectorStore::with_key_prefix`] with the matching prefix
/// so that inserted documents are discoverable by the index.
///
/// # Metadata Fields
///
/// Configure metadata fields via [`RedisVectorStore::with_metadata_fields`] to enable
/// filtering. During insertion, these fields are extracted from the serialized document
/// and stored as separate hash fields that RediSearch can index and filter on.
pub struct RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    model: M,
    connection_manager: ConnectionManager,
    index_name: String,
    vector_field: String,
    key_prefix: Option<String>,
    metadata_fields: Vec<String>,
    distance_metric: DistanceMetric,
}

impl<M> RedisVectorStore<M>
where
    M: EmbeddingModel,
{
    /// Creates a new Redis vector store instance.
    ///
    /// Establishes a [`ConnectionManager`] from the provided client for automatic
    /// reconnection on transient network failures.
    ///
    /// # Arguments
    /// * `model` - Embedding model for query vectorization
    /// * `client` - Redis client instance
    /// * `index_name` - Name of the RediSearch index to query
    /// * `vector_field` - Name of the vector field in the index
    ///
    /// # Errors
    /// Returns an error if the initial connection to Redis cannot be established.
    pub async fn new(
        model: M,
        client: redis::Client,
        index_name: String,
        vector_field: String,
    ) -> Result<Self, VectorStoreError> {
        let connection_manager = ConnectionManager::new(client)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(Self {
            model,
            connection_manager,
            index_name,
            vector_field,
            key_prefix: None,
            metadata_fields: Vec::new(),
            distance_metric: DistanceMetric::default(),
        })
    }

    /// Sets the distance metric the index uses (default [`DistanceMetric::Cosine`]).
    ///
    /// This must match the `DISTANCE_METRIC` of the RediSearch index so that
    /// returned distances are converted to similarity scores correctly. Use
    /// [`Self::validate_index`] to verify the index agrees.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Sets a key prefix for document keys.
    ///
    /// Documents stored via [`InsertDocuments`] will be keyed as `{prefix}{uuid}`.
    /// This prefix **must** match the index's `PREFIX` configuration for documents
    /// to be indexed and discoverable by `FT.SEARCH`.
    pub fn with_key_prefix(mut self, prefix: String) -> Self {
        self.key_prefix = Some(prefix);
        self
    }

    /// Configures metadata fields to extract from documents during insertion.
    ///
    /// When documents are inserted, the specified fields are extracted from the
    /// serialized JSON representation and written as separate hash fields, making
    /// them available for RediSearch filter queries (TAG, NUMERIC, TEXT). The field
    /// names must match top-level keys in the serialized document JSON **and** be
    /// declared in the RediSearch index schema. Calling this method replaces any
    /// previously configured field list.
    ///
    /// Fields that are missing from a document or have null/complex values are
    /// skipped with a warning log. Reserved field names (`document`, `embedded_text`,
    /// and the configured vector field) are filtered out with a warning to prevent
    /// data corruption.
    ///
    /// Note: RediSearch TAG fields split stored values on a separator (`,` by
    /// default). Extracted string values containing the separator will be indexed
    /// as multiple tags; create the TAG field with a different `SEPARATOR` if your
    /// values may contain commas.
    pub fn with_metadata_fields(mut self, fields: Vec<String>) -> Self {
        self.metadata_fields = filter_reserved_metadata_fields(fields, &self.vector_field);
        self
    }

    /// Validates that the configured index exists and is compatible with this store.
    ///
    /// Checks, via `FT.INFO`, that:
    /// - the index exists,
    /// - every vector field uses the store's configured distance metric, and
    /// - if a key prefix is configured, the index is defined with that prefix
    ///   (otherwise inserted documents would never be indexed).
    ///
    /// Call this after building the store to fail fast on schema mismatches.
    pub async fn validate_index(&self) -> Result<(), VectorStoreError> {
        let mut con = self.connection_manager.clone();
        let info: redis::Value = redis::cmd("FT.INFO")
            .arg(&self.index_name)
            .query_async(&mut con)
            .await
            .map_err(|e| {
                VectorStoreError::DatastoreError(
                    format!(
                        "index '{}' not found or FT.INFO failed: {e}",
                        self.index_name
                    )
                    .into(),
                )
            })?;

        let mut tokens = Vec::new();
        Self::flatten_tokens(&info, &mut tokens);

        let expected = self.distance_metric.as_arg();
        for (i, tok) in tokens.iter().enumerate() {
            if tok.eq_ignore_ascii_case("distance_metric") {
                match tokens.get(i + 1) {
                    Some(m) if m.eq_ignore_ascii_case(expected) => {}
                    other => {
                        return Err(VectorStoreError::DatastoreError(
                            format!(
                                "index '{}' uses distance metric {:?}, but this store is configured for {}",
                                self.index_name, other, expected
                            )
                            .into(),
                        ));
                    }
                }
            }
        }

        if let Some(prefix) = &self.key_prefix {
            const STOP: &[&str] = &[
                "default_score",
                "filter",
                "language",
                "language_field",
                "score_field",
                "payload_field",
                "attributes",
            ];
            let found = tokens
                .iter()
                .position(|t| t == "prefixes")
                .map(|p| {
                    tokens[p + 1..]
                        .iter()
                        .take_while(|t| !STOP.contains(&t.as_str()))
                        .any(|t| t == prefix)
                })
                .unwrap_or(false);
            if !found {
                return Err(VectorStoreError::DatastoreError(
                    format!(
                        "index '{}' is not configured with key prefix '{}'",
                        self.index_name, prefix
                    )
                    .into(),
                ));
            }
        }

        Ok(())
    }

    /// Creates the RediSearch index for this store (HASH, `FLAT`, FLOAT32, COSINE).
    ///
    /// Uses the store's index name, vector field, and (if set) key prefix, plus the
    /// `document` and `embedded_text` TEXT fields. Add any metadata fields you intend
    /// to filter on. This is a convenience for setups that manage the index in code;
    /// production deployments may prefer to create the index out of band.
    pub async fn create_index(
        &self,
        dimensions: usize,
        metadata_fields: &[(String, MetadataFieldType)],
    ) -> Result<(), VectorStoreError> {
        let mut con = self.connection_manager.clone();
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg(&self.index_name).arg("ON").arg("HASH");
        if let Some(prefix) = &self.key_prefix {
            cmd.arg("PREFIX").arg(1).arg(prefix);
        }
        cmd.arg("SCHEMA")
            .arg("document")
            .arg("TEXT")
            .arg("embedded_text")
            .arg("TEXT")
            .arg(&self.vector_field)
            .arg("VECTOR")
            .arg("FLAT")
            .arg(6)
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dimensions)
            .arg("DISTANCE_METRIC")
            .arg(self.distance_metric.as_arg());
        for (name, ty) in metadata_fields {
            cmd.arg(name).arg(ty.as_arg());
        }
        cmd.query_async::<()>(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Deletes documents by their hash keys (the IDs returned by [`Self::top_n_ids`]).
    ///
    /// Uses `UNLINK` (non-blocking delete). Returns the number of keys removed.
    pub async fn delete(&self, ids: &[String]) -> Result<u64, VectorStoreError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut con = self.connection_manager.clone();
        let mut cmd = redis::cmd("UNLINK");
        for id in ids {
            cmd.arg(id);
        }
        cmd.query_async::<u64>(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Embeds a query string and returns FLOAT32 LE bytes, rejecting non-finite vectors.
    async fn embed_query(&self, query: &str) -> Result<Vec<u8>, VectorStoreError> {
        let embedding = self.model.embed_text(query).await?;
        if embedding.vec.iter().any(|x| !x.is_finite()) {
            return Err(VectorStoreError::DatastoreError(
                "query embedding contains non-finite (NaN/Inf) values".into(),
            ));
        }
        Ok(Self::embedding_to_bytes(&embedding.vec))
    }

    /// Converts f64 embedding vector to f32 little-endian bytes for Redis VECTOR fields.
    fn embedding_to_bytes(embedding: &[f64]) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect()
    }

    /// Extracts a UTF-8 string from a Redis bulk/simple/verbatim string value.
    fn extract_string(value: &redis::Value) -> Option<String> {
        match value {
            redis::Value::BulkString(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            redis::Value::VerbatimString { text, .. } => Some(text.clone()),
            _ => None,
        }
    }

    /// Parses the raw distance value from a Redis score field.
    ///
    /// The distance is converted to a similarity score by [`DistanceMetric::score`]
    /// according to the store's configured metric.
    fn extract_distance(value: &redis::Value) -> Result<f64, VectorStoreError> {
        let distance = match value {
            redis::Value::Double(d) => *d,
            redis::Value::BulkString(bytes) => {
                String::from_utf8_lossy(bytes).parse::<f64>().map_err(|e| {
                    VectorStoreError::DatastoreError(format!("Failed to parse score: {e}").into())
                })?
            }
            redis::Value::SimpleString(s) | redis::Value::VerbatimString { text: s, .. } => {
                s.parse::<f64>().map_err(|e| {
                    VectorStoreError::DatastoreError(format!("Failed to parse score: {e}").into())
                })?
            }
            other => {
                return Err(VectorStoreError::DatastoreError(
                    format!("Unexpected Redis value type for score: {other:?}").into(),
                ));
            }
        };
        Ok(distance)
    }

    /// Parses an FT.SEARCH response into results with deserialized documents.
    ///
    /// Documents with empty or unparseable JSON are skipped with a warning rather
    /// than aborting the entire result set.
    fn parse_search_response<T>(
        response: redis::Value,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        Self::parse_response_generic(response, true).map(|items| {
            items
                .into_iter()
                .filter_map(|(score, id, doc_json)| {
                    if doc_json.is_empty() {
                        tracing::warn!(
                            target: "rig",
                            id = %id,
                            "Document field missing or empty in hash, skipping"
                        );
                        return None;
                    }
                    match serde_json::from_str::<T>(&doc_json) {
                        Ok(doc) => Some((score, id, doc)),
                        Err(e) => {
                            tracing::warn!(
                                target: "rig",
                                id = %id,
                                error = %e,
                                "Failed to deserialize document, skipping"
                            );
                            None
                        }
                    }
                })
                .collect()
        })
    }

    /// Parses an FT.SEARCH response for IDs and scores only.
    fn parse_search_response_ids(
        response: redis::Value,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Self::parse_response_generic(response, false).map(|items| {
            items
                .into_iter()
                .map(|(score, id, _)| (score, id))
                .collect()
        })
    }

    /// Generic response parser handling both RESP2 (array) and RESP3 (map)
    /// `FT.SEARCH` reply shapes, in full-document or ID-only modes.
    fn parse_response_generic(
        response: redis::Value,
        include_document: bool,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        match response {
            // RESP3: a map with "results" => [ {id, extra_attributes: {..}}, .. ].
            redis::Value::Map(pairs) => Self::parse_resp3_map(&pairs, include_document),
            // RESP2: [count, key1, [field, val, ..], key2, [..], ..].
            redis::Value::Array(items) => Self::parse_resp2_array(&items, include_document),
            _ => Err(VectorStoreError::DatastoreError(
                "Invalid FT.SEARCH response format (expected a RESP2 array or RESP3 map)".into(),
            )),
        }
    }

    /// Parses the RESP2 flat-array `FT.SEARCH` reply.
    fn parse_resp2_array(
        items: &[redis::Value],
        include_document: bool,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        let count = match items.first() {
            Some(redis::Value::Int(n)) => *n as usize,
            _ => {
                return Err(VectorStoreError::DatastoreError(
                    "Invalid response format: expected count as first element".into(),
                ));
            }
        };

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(count);

        let mut iter = items.iter().skip(1);
        while let Some(key_val) = iter.next() {
            let id = match Self::extract_string(key_val) {
                Some(id) => id,
                None => {
                    iter.next();
                    continue;
                }
            };

            let fields_val = match iter.next() {
                Some(redis::Value::Array(fields)) => fields,
                _ => continue,
            };

            let mut distance = 0.0;
            let mut score_found = false;
            let mut document_json = String::new();

            for chunk in fields_val.chunks(2) {
                let [name_val, value_val] = chunk else {
                    continue;
                };
                let field_name = match Self::extract_string(name_val) {
                    Some(name) => name,
                    None => continue,
                };

                if field_name == "__vector_score" {
                    distance = Self::extract_distance(value_val)?;
                    score_found = true;
                } else if include_document && field_name == "document" {
                    match Self::extract_string(value_val) {
                        Some(json) => document_json = json,
                        None => {
                            tracing::warn!(
                                target: "rig",
                                id = %id,
                                "Document field present but could not be extracted as string"
                            );
                        }
                    }
                }
            }

            if !score_found {
                tracing::warn!(
                    target: "rig",
                    id = %id,
                    "__vector_score field missing from search result, defaulting to 0.0"
                );
            }

            results.push((distance, id, document_json));
        }

        Ok(results)
    }

    /// Parses the RESP3 map-shaped `FT.SEARCH` reply.
    fn parse_resp3_map(
        pairs: &[(redis::Value, redis::Value)],
        include_document: bool,
    ) -> Result<Vec<(f64, String, String)>, VectorStoreError> {
        let entries = pairs
            .iter()
            .find_map(|(k, v)| match (Self::extract_string(k), v) {
                (Some(name), redis::Value::Array(items)) if name == "results" => Some(items),
                _ => None,
            });

        let Some(entries) = entries else {
            // No "results" key (e.g. total_results 0) -> no matches.
            return Ok(Vec::new());
        };

        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            let redis::Value::Map(fields) = entry else {
                continue;
            };

            let mut id = String::new();
            let mut distance = 0.0;
            let mut score_found = false;
            let mut document_json = String::new();

            for (k, v) in fields {
                match Self::extract_string(k).as_deref() {
                    Some("id") => {
                        if let Some(s) = Self::extract_string(v) {
                            id = s;
                        }
                    }
                    Some("extra_attributes") => {
                        if let redis::Value::Map(attrs) = v {
                            for (ak, av) in attrs {
                                match Self::extract_string(ak).as_deref() {
                                    Some("__vector_score") => {
                                        distance = Self::extract_distance(av)?;
                                        score_found = true;
                                    }
                                    Some("document") if include_document => {
                                        if let Some(s) = Self::extract_string(av) {
                                            document_json = s;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            if !score_found {
                tracing::warn!(
                    target: "rig",
                    id = %id,
                    "__vector_score field missing from search result, defaulting to 0.0"
                );
            }

            results.push((distance, id, document_json));
        }

        Ok(results)
    }

    /// Recursively flattens a Redis reply into its scalar string tokens, in order.
    /// Used to inspect `FT.INFO` output without depending on its exact shape.
    fn flatten_tokens(value: &redis::Value, out: &mut Vec<String>) {
        match value {
            redis::Value::Array(items) | redis::Value::Set(items) => {
                for v in items {
                    Self::flatten_tokens(v, out);
                }
            }
            redis::Value::Map(pairs) => {
                for (k, v) in pairs {
                    Self::flatten_tokens(k, out);
                    Self::flatten_tokens(v, out);
                }
            }
            redis::Value::BulkString(bytes) => out.push(String::from_utf8_lossy(bytes).to_string()),
            redis::Value::SimpleString(s) => out.push(s.clone()),
            redis::Value::VerbatimString { text, .. } => out.push(text.clone()),
            redis::Value::Int(i) => out.push(i.to_string()),
            redis::Value::Double(d) => out.push(d.to_string()),
            _ => {}
        }
    }

    /// Builds and executes an FT.SEARCH KNN query.
    async fn execute_search(
        &self,
        vector_bytes: Vec<u8>,
        req: &VectorSearchRequest<Filter>,
        include_document: bool,
    ) -> Result<redis::Value, VectorStoreError> {
        let mut con = self.connection_manager.clone();

        let filter_str = req
            .filter()
            .as_ref()
            .map(|f| f.clone().into_inner())
            .unwrap_or_else(|| "*".to_string());

        let knn_query = format!(
            "{}=>[KNN {} @{} $vec AS __vector_score]",
            filter_str,
            req.samples(),
            self.vector_field
        );

        let mut cmd = redis::cmd("FT.SEARCH");
        cmd.arg(&self.index_name)
            .arg(&knn_query)
            .arg("PARAMS")
            .arg(2)
            .arg("vec")
            .arg(vector_bytes)
            .arg("SORTBY")
            .arg("__vector_score")
            .arg("RETURN");

        if include_document {
            cmd.arg(2).arg("__vector_score").arg("document");
        } else {
            cmd.arg(1).arg("__vector_score");
        }

        cmd.arg("DIALECT").arg(2);

        // Always specify LIMIT to override RediSearch's default of 10 results.
        cmd.arg("LIMIT").arg(0).arg(req.samples());

        cmd.query_async(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
    }

    /// Converts a JSON value to a string suitable for a flat Redis hash field.
    ///
    /// Strings are stored unquoted, numbers/booleans use their string form
    /// (`1`/`0` for booleans). Null/array/object return `None`.
    fn json_value_to_hash_field(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(if *b { "1".to_string() } else { "0".to_string() }),
            serde_json::Value::Null
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_) => None,
        }
    }
}

impl<Model> InsertDocuments for RedisVectorStore<Model>
where
    Model: EmbeddingModel + Send + Sync,
{
    /// Inserts documents with their precomputed embeddings into Redis.
    ///
    /// Each embedding in [`OneOrMany<Embedding>`] produces a separate Redis hash
    /// keyed by `{prefix}{uuid}`. All hashes for a document share the same serialized
    /// JSON in the `document` field but have distinct `embedded_text` values.
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        let mut con = self.connection_manager.clone();
        let mut pipe = redis::pipe();

        for (document, embeddings) in &documents {
            let json_value = serde_json::to_value(document)?;
            let json_document = json_value.to_string();

            // Extract configured metadata fields from the document JSON.
            let metadata: Vec<(String, String)> = if self.metadata_fields.is_empty() {
                Vec::new()
            } else {
                self.metadata_fields
                    .iter()
                    .filter_map(|field_name| {
                        let value = json_value.get(field_name)?;
                        match Self::json_value_to_hash_field(value) {
                            Some(hash_value) => Some((field_name.clone(), hash_value)),
                            None => {
                                tracing::warn!(
                                    target: "rig",
                                    field = %field_name,
                                    value_type = %value,
                                    "Metadata field has unsupported type (null/array/object), skipping"
                                );
                                None
                            }
                        }
                    })
                    .collect()
            };

            for embedding in embeddings.iter() {
                let id = if let Some(ref prefix) = self.key_prefix {
                    format!("{}{}", prefix, uuid::Uuid::new_v4())
                } else {
                    uuid::Uuid::new_v4().to_string()
                };
                let embedding_bytes = Self::embedding_to_bytes(&embedding.vec);

                let cmd = pipe
                    .cmd("HSET")
                    .arg(&id)
                    .arg("document")
                    .arg(json_document.as_bytes())
                    .arg("embedded_text")
                    .arg(embedding.document.as_bytes())
                    .arg(&self.vector_field)
                    .arg(embedding_bytes);

                for (field_name, field_value) in &metadata {
                    cmd.arg(field_name).arg(field_value.as_bytes());
                }

                cmd.ignore();
            }
        }

        pipe.query_async::<()>(&mut con)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        tracing::debug!(
            target: "rig",
            index = %self.index_name,
            count = documents.len(),
            metadata_fields = ?self.metadata_fields,
            "Inserted documents into Redis vector store"
        );

        Ok(())
    }
}

impl<M> VectorStoreIndex for RedisVectorStore<M>
where
    M: EmbeddingModel + Send + Sync,
{
    type Filter = Filter;

    async fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        if req.samples() == 0 {
            return Ok(Vec::new());
        }
        let vector_bytes = self.embed_query(req.query()).await?;

        let response = self.execute_search(vector_bytes, &req, true).await?;
        let mut results = Self::parse_search_response::<T>(response)?
            .into_iter()
            .map(|(distance, id, doc)| (self.distance_metric.score(distance), id, doc))
            .collect::<Vec<_>>();

        if let Some(threshold) = req.threshold() {
            results.retain(|(score, _, _)| *score >= threshold);
        }

        tracing::debug!(
            target: "rig",
            index = %self.index_name,
            query = %req.query(),
            "Selected documents: {}",
            results.iter().map(|(score, id, _)| format!("{id} ({score:.4})")).collect::<Vec<_>>().join(", ")
        );

        Ok(results)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        if req.samples() == 0 {
            return Ok(Vec::new());
        }
        let vector_bytes = self.embed_query(req.query()).await?;

        let response = self.execute_search(vector_bytes, &req, false).await?;
        let mut results = Self::parse_search_response_ids(response)?
            .into_iter()
            .map(|(distance, id)| (self.distance_metric.score(distance), id))
            .collect::<Vec<_>>();

        if let Some(threshold) = req.threshold() {
            results.retain(|(score, _)| *score >= threshold);
        }

        tracing::debug!(
            target: "rig",
            index = %self.index_name,
            query = %req.query(),
            "Selected document IDs: {}",
            results.iter().map(|(score, id)| format!("{id} ({score:.4})")).collect::<Vec<_>>().join(", ")
        );

        Ok(results)
    }
}

impl<M> VectorStoreIndexDyn for RedisVectorStore<M>
where
    M: EmbeddingModel + Sync + Send,
{
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<CoreFilter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults> {
        Box::pin(async move {
            let req = req.try_map_filter(Filter::try_from)?;
            let results = <Self as VectorStoreIndex>::top_n::<serde_json::Value>(self, req).await?;
            Ok(results)
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<CoreFilter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        Box::pin(async move {
            let req = req.try_map_filter(Filter::try_from)?;
            let results = <Self as VectorStoreIndex>::top_n_ids(self, req).await?;
            Ok(results)
        })
    }
}

/// Filters out reserved hash field names (`document`, `embedded_text`, and the
/// vector field) from a configured metadata field list, emitting a warning for
/// each removed name to prevent overwriting reserved hash fields.
fn filter_reserved_metadata_fields(fields: Vec<String>, vector_field: &str) -> Vec<String> {
    let reserved = ["document", "embedded_text", vector_field];
    fields
        .into_iter()
        .filter(|f| {
            if reserved.contains(&f.as_str()) {
                tracing::warn!(
                    target: "rig",
                    field = %f,
                    "Metadata field name conflicts with reserved hash field, skipping"
                );
                false
            } else {
                true
            }
        })
        .collect()
}

/// RediSearch vector distance metric. Determines how the returned distance is
/// converted to a similarity score (higher = more similar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine distance. Score = `1 - distance` (1 = identical, -1 = opposite).
    #[default]
    Cosine,
    /// Squared Euclidean (L2) distance in `[0, inf)`. Score = `1 / (1 + distance)`
    /// (1 = identical, approaching 0 as vectors get farther apart).
    L2,
    /// Inner-product distance. RediSearch returns `1 - inner_product`, so
    /// Score = `1 - distance` (equal to the inner product; higher = more similar).
    InnerProduct,
}

impl DistanceMetric {
    /// The `DISTANCE_METRIC` argument value for `FT.CREATE`.
    fn as_arg(self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "COSINE",
            DistanceMetric::L2 => "L2",
            DistanceMetric::InnerProduct => "IP",
        }
    }

    /// Converts a RediSearch distance into a similarity score where higher means
    /// more similar. The conversion is monotonically decreasing in `distance`, so
    /// it preserves RediSearch's nearest-first ordering for every metric.
    fn score(self, distance: f64) -> f64 {
        match self {
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => 1.0 - distance,
            DistanceMetric::L2 => 1.0 / (1.0 + distance),
        }
    }
}

/// RediSearch field type for a metadata field declared via [`RedisVectorStore::create_index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataFieldType {
    /// Exact-match tag field (`@field:{value}`).
    Tag,
    /// Numeric field supporting range filters (`@field:[min max]`).
    Numeric,
    /// Full-text field (`@field:(tokens)`).
    Text,
}

impl MetadataFieldType {
    fn as_arg(self) -> &'static str {
        match self {
            MetadataFieldType::Tag => "TAG",
            MetadataFieldType::Numeric => "NUMERIC",
            MetadataFieldType::Text => "TEXT",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::embeddings::embedding::EmbeddingError;

    /// Minimal embedding model used only to name a concrete `RedisVectorStore`
    /// type for calling its self-less associated helpers in unit tests.
    struct FakeModel;

    impl EmbeddingModel for FakeModel {
        const MAX_DOCUMENTS: usize = 1024;
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            FakeModel
        }

        fn ndims(&self) -> usize {
            3
        }

        async fn embed_texts(
            &self,
            _texts: impl IntoIterator<Item = String> + Send,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Ok(Vec::new())
        }
    }

    type Store = RedisVectorStore<FakeModel>;

    fn bulk(s: &str) -> redis::Value {
        redis::Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn reserved_metadata_fields_are_filtered() {
        let kept = filter_reserved_metadata_fields(
            vec![
                "category".to_string(),
                "document".to_string(),
                "embedded_text".to_string(),
                "embedding".to_string(),
                "price".to_string(),
            ],
            "embedding",
        );
        assert_eq!(kept, vec!["category".to_string(), "price".to_string()]);
    }

    #[test]
    fn json_value_to_hash_field_covers_all_types() {
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!("hello")),
            Some("hello".to_string())
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!(3)),
            Some("3".to_string())
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!(true)),
            Some("1".to_string())
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!(false)),
            Some("0".to_string())
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::Value::Null),
            None
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!([1, 2])),
            None
        );
        assert_eq!(
            Store::json_value_to_hash_field(&serde_json::json!({"a": 1})),
            None
        );
    }

    #[test]
    fn embedding_to_bytes_is_float32_le() {
        let bytes = Store::embedding_to_bytes(&[1.0_f64]);
        assert_eq!(bytes, vec![0, 0, 128, 63]); // 1.0_f32 little-endian
    }

    #[test]
    fn parse_search_response_skips_empty_documents() {
        // count=2: doc:1 has valid JSON, doc:2 has an empty document field.
        let response = redis::Value::Array(vec![
            redis::Value::Int(2),
            bulk("doc:1"),
            redis::Value::Array(vec![
                bulk("__vector_score"),
                bulk("0.1"),
                bulk("document"),
                bulk("{\"a\":1}"),
            ]),
            bulk("doc:2"),
            redis::Value::Array(vec![
                bulk("__vector_score"),
                bulk("0.2"),
                bulk("document"),
                bulk(""),
            ]),
        ]);

        let results =
            Store::parse_search_response::<serde_json::Value>(response).expect("parse ok");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "doc:1");
        assert!((results[0].0 - 0.1).abs() < 1e-9); // raw distance, converted by the metric later
    }

    #[test]
    fn parse_search_response_empty_when_count_zero() {
        let response = redis::Value::Array(vec![redis::Value::Int(0)]);
        let results =
            Store::parse_search_response::<serde_json::Value>(response).expect("parse ok");
        assert!(results.is_empty());
    }

    #[test]
    fn parse_resp3_map_response() {
        // RESP3 FT.SEARCH reply shape: a map with a "results" array of per-doc maps.
        let response = redis::Value::Map(vec![
            (bulk("attributes"), redis::Value::Array(vec![])),
            (bulk("format"), bulk("STRING")),
            (
                bulk("results"),
                redis::Value::Array(vec![redis::Value::Map(vec![
                    (bulk("id"), bulk("d:1")),
                    (
                        bulk("extra_attributes"),
                        redis::Value::Map(vec![
                            (bulk("__vector_score"), bulk("0.1")),
                            (bulk("document"), bulk("{\"a\":1}")),
                        ]),
                    ),
                ])]),
            ),
            (bulk("total_results"), redis::Value::Int(1)),
        ]);

        let results =
            Store::parse_search_response::<serde_json::Value>(response).expect("parse ok");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "d:1");
        assert!((results[0].0 - 0.1).abs() < 1e-9); // raw distance
    }

    #[test]
    fn parse_resp3_map_empty_results() {
        let response = redis::Value::Map(vec![
            (bulk("results"), redis::Value::Array(vec![])),
            (bulk("total_results"), redis::Value::Int(0)),
        ]);
        let results =
            Store::parse_search_response::<serde_json::Value>(response).expect("parse ok");
        assert!(results.is_empty());
    }

    #[test]
    fn distance_metric_score_conversions() {
        // Cosine: 1 - distance, range [-1, 1].
        assert!((DistanceMetric::Cosine.score(0.0) - 1.0).abs() < 1e-9);
        assert!((DistanceMetric::Cosine.score(2.0) - (-1.0)).abs() < 1e-9);
        // Inner product: 1 - distance (== the dot product).
        assert!((DistanceMetric::InnerProduct.score(0.0) - 1.0).abs() < 1e-9);
        assert!((DistanceMetric::InnerProduct.score(0.5) - 0.5).abs() < 1e-9);
        // L2: 1 / (1 + distance), range (0, 1].
        assert!((DistanceMetric::L2.score(0.0) - 1.0).abs() < 1e-9);
        assert!((DistanceMetric::L2.score(3.0) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn distance_metric_score_is_monotonic_decreasing() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::L2,
            DistanceMetric::InnerProduct,
        ] {
            assert!(
                metric.score(0.1) > metric.score(0.5),
                "{metric:?} score must decrease as distance grows"
            );
        }
    }

    #[test]
    fn distance_metric_as_arg() {
        assert_eq!(DistanceMetric::Cosine.as_arg(), "COSINE");
        assert_eq!(DistanceMetric::L2.as_arg(), "L2");
        assert_eq!(DistanceMetric::InnerProduct.as_arg(), "IP");
    }
}
