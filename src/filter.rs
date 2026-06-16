//! Redis-specific filter types for RediSearch `FT.SEARCH` queries.
//!
//! Provides [`Filter`] which implements [`SearchFilter`] and translates
//! Rig's generic filter expressions into RediSearch query syntax.
//!
//! # Field Type Expectations
//!
//! - **Numeric fields**: `eq`, `gt`, `lt`, `gte`, `lte`, and `range` produce range syntax (`@field:[min max]`)
//! - **Tag fields**: String equality uses tag syntax (`@field:{value}`)
//! - **Bool fields**: Equality uses numeric TAG values (`1` or `0`) with tag syntax
//! - **Range filters**: Redis-specific range constructors reject non-numeric values instead of converting them into tag matches
//!
//! The generic [`SearchFilter`] implementation is numeric-only because Rig's
//! shared trait is infallible. Use the inherent [`Filter`] constructors when
//! building Redis filters directly, especially for tag values and fallible
//! numeric range filters. Ensure your RediSearch schema matches the filter
//! types you use.
//!
//! # Dynamic API Subset
//!
//! The [`VectorStoreIndexDyn`](rig_core::vector_store::VectorStoreIndexDyn) path
//! only supports the five [`CoreFilter`](rig_core::vector_store::request::Filter)
//! variants: `Eq`, `Gt`, `Lt`, `And`, `Or`. Redis-specific methods such as
//! [`Filter::gte`], [`Filter::lte`], [`Filter::range`], [`Filter::tag_in`],
//! [`Filter::text_contains`], and [`Filter::text_phrase`] are only reachable
//! through the typed [`VectorStoreIndex`](rig_core::vector_store::VectorStoreIndex) API.
//!
//! # Escaping
//!
//! Field names and tag/text values are escaped for RediSearch special characters
//! by the typed constructors. The [`escape_tag_value`] and [`escape_text_value`]
//! helpers are exported for callers building raw queries via [`Filter::raw`].

use rig_core::vector_store::request::{Filter as CoreFilter, FilterError, SearchFilter};
use serde::{Deserialize, Serialize};

/// Typed value for Redis-specific filter expressions.
///
/// Determines how the value is formatted in the RediSearch query syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum RedisValue {
    /// Numeric value. Equality produces range syntax; range constructors use
    /// this as the only supported value kind.
    Number(f64),
    /// String/tag value. Equality and tag filters produce tag syntax.
    String(String),
    /// Boolean value. Equality treats this as a TAG value (`1` or `0`).
    Bool(bool),
}

/// Finite numeric value for Redis range-syntax filters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedisNumber(f64);

impl RedisNumber {
    /// Creates a finite Redis numeric filter value.
    ///
    /// Returns a [`FilterError`] if `value` is not finite (`NaN`, `+inf`, `-inf`).
    pub fn new(value: f64) -> Result<Self, FilterError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(FilterError::Expected {
                expected: "finite numeric value for Redis numeric filter".into(),
                got: value.to_string(),
            })
        }
    }

    fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for RedisNumber {
    type Error = FilterError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Note: `i64` values with magnitude above 2^53 lose precision in the `f64`
/// representation used by RediSearch numeric fields.
impl From<i64> for RedisNumber {
    fn from(value: i64) -> Self {
        Self(value as f64)
    }
}

/// Note: `u64` values above 2^53 (9,007,199,254,740,992) lose precision in the
/// `f64` representation used by RediSearch numeric fields.
impl From<u64> for RedisNumber {
    fn from(value: u64) -> Self {
        Self(value as f64)
    }
}

fn numeric_bound(value: RedisValue, operation: &'static str) -> Result<RedisNumber, FilterError> {
    match value {
        RedisValue::Number(n) if n.is_finite() => Ok(RedisNumber(n)),
        RedisValue::Number(n) => Err(FilterError::Expected {
            expected: format!("finite numeric value for Redis {operation} filter"),
            got: n.to_string(),
        }),
        other => Err(FilterError::Expected {
            expected: format!("numeric value for Redis {operation} filter"),
            got: format!("{other:?}"),
        }),
    }
}

fn numeric_eq_filter(key: impl AsRef<str>, value: RedisNumber) -> Filter {
    let value = value.get();
    Filter(format!("@{}:[{value} {value}]", field_name(key)))
}

fn gt_number_filter(key: impl AsRef<str>, value: RedisNumber) -> Filter {
    let value = value.get();
    Filter(format!("@{}:[({value} +inf]", field_name(key)))
}

fn lt_number_filter(key: impl AsRef<str>, value: RedisNumber) -> Filter {
    let value = value.get();
    Filter(format!("@{}:[-inf ({value}]", field_name(key)))
}

fn gte_number_filter(key: impl AsRef<str>, value: RedisNumber) -> Filter {
    let value = value.get();
    Filter(format!("@{}:[{value} +inf]", field_name(key)))
}

fn lte_number_filter(key: impl AsRef<str>, value: RedisNumber) -> Filter {
    let value = value.get();
    Filter(format!("@{}:[-inf {value}]", field_name(key)))
}

fn field_name(key: impl AsRef<str>) -> String {
    key.as_ref()
        .split('.')
        .map(escape_field_segment)
        .collect::<Vec<_>>()
        .join(".")
}

fn escape_field_segment(segment: &str) -> String {
    let mut escaped = String::with_capacity(segment.len());
    for ch in segment.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            escaped.push(ch);
        } else {
            escaped.push('\\');
            escaped.push(ch);
        }
    }
    escaped
}

/// Escapes RediSearch tag special characters in `value` by backslash-prefixing them.
///
/// Useful when constructing raw queries with [`Filter::raw`]. The typed
/// constructors apply this automatically.
pub fn escape_tag_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | ' '
                | ','
                | '.'
                | '<'
                | '>'
                | '{'
                | '}'
                | '['
                | ']'
                | '"'
                | '\''
                | ':'
                | ';'
                | '!'
                | '@'
                | '#'
                | '$'
                | '%'
                | '^'
                | '&'
                | '*'
                | '('
                | ')'
                | '-'
                | '+'
                | '='
                | '~'
                | '|'
                | '/'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

/// Escapes RediSearch TEXT query special characters in `value` by backslash-prefixing them.
///
/// Useful when constructing raw queries with [`Filter::raw`]. The typed
/// text constructors ([`Filter::text_contains`], [`Filter::text_phrase`]) apply
/// this automatically.
pub fn escape_text_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '<'
                | '>'
                | '{'
                | '}'
                | '['
                | ']'
                | '"'
                | '\''
                | ':'
                | ';'
                | '!'
                | '@'
                | '#'
                | '$'
                | '%'
                | '^'
                | '&'
                | '*'
                | '('
                | ')'
                | '-'
                | '+'
                | '='
                | '~'
                | '|'
                | '/'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

impl From<i64> for RedisValue {
    fn from(value: i64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<u64> for RedisValue {
    fn from(value: u64) -> Self {
        Self::Number(value as f64)
    }
}

impl From<f64> for RedisValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for RedisValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<String> for RedisValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for RedisValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl TryFrom<serde_json::Value> for RedisValue {
    type Error = FilterError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::Bool(b) => Ok(RedisValue::Bool(b)),
            serde_json::Value::Number(n) => {
                let num = n.as_f64().ok_or_else(|| FilterError::Expected {
                    expected: "Valid 64-bit float".into(),
                    got: "Invalid 64-bit float".into(),
                })?;
                Ok(RedisValue::Number(num))
            }
            serde_json::Value::String(s) => Ok(RedisValue::String(s)),
            serde_json::Value::Null
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_) => Err(FilterError::TypeError(
                "Redis filter does not currently support null values, arrays or objects".into(),
            )),
        }
    }
}

/// Redis filter for FT.SEARCH queries.
///
/// Wraps a raw RediSearch query string. Use the inherent constructors on this
/// type for Redis-specific filters, including fallible numeric range filters
/// and tag filters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter(String);

impl SearchFilter for Filter {
    type Value = RedisNumber;

    /// Numeric equality filter.
    ///
    /// Redis-specific string and boolean equality are available through the
    /// inherent [`Filter::eq`] constructor.
    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        numeric_eq_filter(key, value)
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        gt_number_filter(key, value)
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        lt_number_filter(key, value)
    }

    fn and(self, rhs: Self) -> Self {
        self.and(rhs)
    }

    fn or(self, rhs: Self) -> Self {
        self.or(rhs)
    }
}

impl Filter {
    /// Equality filter.
    ///
    /// - Numeric: `@field:[val val]` (exact range match)
    /// - String: `@field:{value}` (tag match, value escaped)
    /// - Bool: `@field:{1}` or `@field:{0}` (tag match)
    pub fn eq(key: impl AsRef<str>, value: impl Into<RedisValue>) -> Result<Self, FilterError> {
        let key = field_name(key);
        let filter = match value.into() {
            RedisValue::Number(n) => numeric_eq_filter(key, RedisNumber::new(n)?),
            RedisValue::String(ref s) => Self(format!("@{key}:{{{}}}", escape_tag_value(s))),
            RedisValue::Bool(b) => {
                let v = if b { "1" } else { "0" };
                Self(format!("@{key}:{{{v}}}"))
            }
        };
        Ok(filter)
    }

    /// Greater-than filter (exclusive).
    ///
    /// Produces `@field:[(val +inf]` for numeric values and returns an error
    /// for strings or booleans. Use [`Filter::eq`] or [`Filter::tag_in`] for
    /// tag comparisons.
    pub fn gt(key: impl AsRef<str>, value: impl Into<RedisValue>) -> Result<Self, FilterError> {
        let value = numeric_bound(value.into(), "greater-than")?;
        Ok(gt_number_filter(key, value))
    }

    /// Less-than filter (exclusive).
    ///
    /// Produces `@field:[-inf (val]` for numeric values and returns an error
    /// for strings or booleans. Use [`Filter::eq`] or [`Filter::tag_in`] for
    /// tag comparisons.
    pub fn lt(key: impl AsRef<str>, value: impl Into<RedisValue>) -> Result<Self, FilterError> {
        let value = numeric_bound(value.into(), "less-than")?;
        Ok(lt_number_filter(key, value))
    }

    /// Negates this filter expression.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("-{}", self.0))
    }

    /// Greater than or equal (inclusive).
    ///
    /// Produces `@field:[val +inf]` for numeric values and returns an error
    /// for strings or booleans. Use [`Filter::eq`] or [`Filter::tag_in`] for
    /// tag comparisons.
    pub fn gte(key: impl AsRef<str>, value: impl Into<RedisValue>) -> Result<Self, FilterError> {
        let value = numeric_bound(value.into(), "greater-than-or-equal")?;
        Ok(gte_number_filter(key, value))
    }

    /// Less than or equal (inclusive).
    ///
    /// Produces `@field:[-inf val]` for numeric values and returns an error
    /// for strings or booleans. Use [`Filter::eq`] or [`Filter::tag_in`] for
    /// tag comparisons.
    pub fn lte(key: impl AsRef<str>, value: impl Into<RedisValue>) -> Result<Self, FilterError> {
        let value = numeric_bound(value.into(), "less-than-or-equal")?;
        Ok(lte_number_filter(key, value))
    }

    /// Combines this filter with another using RediSearch AND semantics.
    pub fn and(self, rhs: Self) -> Self {
        Self(format!("({} {})", self.0, rhs.0))
    }

    /// Combines this filter with another using RediSearch OR semantics.
    pub fn or(self, rhs: Self) -> Self {
        Self(format!("({} | {})", self.0, rhs.0))
    }

    /// Numeric range filter (inclusive on both ends).
    pub fn range(key: impl AsRef<str>, min: f64, max: f64) -> Result<Self, FilterError> {
        let min = RedisNumber::new(min)?.get();
        let max = RedisNumber::new(max)?.get();
        Ok(Self(format!("@{}:[{} {}]", field_name(key), min, max)))
    }

    /// Numeric range filter (exclusive on both ends).
    pub fn range_exclusive(key: impl AsRef<str>, min: f64, max: f64) -> Result<Self, FilterError> {
        let min = RedisNumber::new(min)?.get();
        let max = RedisNumber::new(max)?.get();
        Ok(Self(format!("@{}:[({} ({}]", field_name(key), min, max)))
    }

    /// Tag filter for multiple values (OR).
    ///
    /// Produces `@field:{val1 | val2 | val3}` with each value escaped.
    ///
    /// If `values` is empty, returns a match-all filter (`*`) to avoid emitting
    /// invalid `@field:{}` syntax.
    pub fn tag_in(key: impl AsRef<str>, values: Vec<String>) -> Self {
        if values.is_empty() {
            return Self::raw("*");
        }
        let tags = values
            .iter()
            .map(|value| escape_tag_value(value))
            .collect::<Vec<_>>()
            .join(" | ");
        Self(format!("@{}:{{{}}}", field_name(key), tags))
    }

    /// Full-text token search within a TEXT field.
    ///
    /// Performs a token-AND search: all words in `text` must appear in the field,
    /// in any order or position. This is **not** a phrase search; use
    /// [`Filter::text_phrase`] for exact ordered matching.
    pub fn text_contains(key: impl AsRef<str>, text: impl AsRef<str>) -> Self {
        Self(format!(
            "@{}:({})",
            field_name(key),
            escape_text_value(text.as_ref())
        ))
    }

    /// Exact phrase search within a TEXT field.
    ///
    /// Matches the exact ordered sequence of words in `phrase`. Produces
    /// `@field:"phrase"` syntax with the phrase contents escaped (the wrapping
    /// quotes are not escaped).
    pub fn text_phrase(key: impl AsRef<str>, phrase: impl AsRef<str>) -> Self {
        Self(format!(
            "@{}:\"{}\"",
            field_name(key),
            escape_text_value(phrase.as_ref())
        ))
    }

    /// Creates a filter from a raw RediSearch query string.
    ///
    /// No escaping is applied; the caller is responsible for valid syntax.
    ///
    /// # Security Warning
    ///
    /// Do **not** pass unsanitized user input to this method. Arbitrary
    /// RediSearch clauses can be injected, returning unintended results or
    /// causing parse errors. Use the typed constructors (e.g. [`Filter::eq`],
    /// [`Filter::tag_in`]) for user-supplied values, or escape values yourself
    /// with [`escape_tag_value`] / [`escape_text_value`].
    pub fn raw(query: impl Into<String>) -> Self {
        Self(query.into())
    }

    /// Consumes the filter and returns the raw RediSearch query string.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl TryFrom<CoreFilter<serde_json::Value>> for Filter {
    type Error = FilterError;

    fn try_from(value: CoreFilter<serde_json::Value>) -> Result<Self, Self::Error> {
        let filter = match value {
            CoreFilter::Eq(k, val) => Filter::eq(k, RedisValue::try_from(val)?)?,
            CoreFilter::Gt(k, val) => Filter::gt(k, RedisValue::try_from(val)?)?,
            CoreFilter::Lt(k, val) => Filter::lt(k, RedisValue::try_from(val)?)?,
            CoreFilter::And(l, r) => Self::try_from(*l)?.and(Self::try_from(*r)?),
            CoreFilter::Or(l, r) => Self::try_from(*l)?.or(Self::try_from(*r)?),
        };

        Ok(filter)
    }
}
