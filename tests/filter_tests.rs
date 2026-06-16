//! Unit tests for the public RediSearch filter API.

use rig_core::vector_store::request::Filter as CoreFilter;
use rig_redis_vectorstore::Filter;
use rig_redis_vectorstore::filter::{RedisNumber, escape_tag_value, escape_text_value};

fn rendered(f: Filter) -> String {
    f.into_inner()
}

#[test]
fn numeric_equality_uses_range_syntax() {
    let f = Filter::eq("age", 30_i64).unwrap();
    assert_eq!(rendered(f), "@age:[30 30]");
}

#[test]
fn string_equality_uses_tag_syntax() {
    let f = Filter::eq("category", "Electronics").unwrap();
    assert_eq!(rendered(f), "@category:{Electronics}");
}

#[test]
fn bool_equality_uses_tag_one_zero() {
    assert_eq!(rendered(Filter::eq("active", true).unwrap()), "@active:{1}");
    assert_eq!(
        rendered(Filter::eq("active", false).unwrap()),
        "@active:{0}"
    );
}

#[test]
fn greater_than_numeric() {
    let f = Filter::gt("price", 100_i64).unwrap();
    assert_eq!(rendered(f), "@price:[(100 +inf]");
}

#[test]
fn range_constructors() {
    assert_eq!(
        rendered(Filter::range("p", 1.0, 10.0).unwrap()),
        "@p:[1 10]"
    );
    assert_eq!(
        rendered(Filter::range_exclusive("p", 1.0, 10.0).unwrap()),
        "@p:[(1 (10]"
    );
}

#[test]
fn range_ops_reject_non_numeric() {
    assert!(Filter::gt("status", "active").is_err());
    assert!(Filter::lt("status", "active").is_err());
    assert!(Filter::gte("status", true).is_err());
    assert!(Filter::lte("status", "x").is_err());
}

#[test]
fn tag_in_joins_values_and_guards_empty() {
    assert_eq!(
        rendered(Filter::tag_in("cat", vec!["a".into(), "b".into()])),
        "@cat:{a | b}"
    );
    assert_eq!(rendered(Filter::tag_in("cat", vec![])), "*");
}

#[test]
fn text_contains_and_phrase() {
    assert_eq!(
        rendered(Filter::text_contains("desc", "hello world")),
        "@desc:(hello world)"
    );
    assert_eq!(
        rendered(Filter::text_phrase("desc", "hello world")),
        "@desc:\"hello world\""
    );
}

#[test]
fn negation_and_combination() {
    let f = Filter::eq("a", 1_i64).unwrap().not();
    assert_eq!(rendered(f), "-@a:[1 1]");

    let combined = Filter::eq("a", 1_i64)
        .unwrap()
        .and(Filter::eq("b", 2_i64).unwrap());
    assert_eq!(rendered(combined), "(@a:[1 1] @b:[2 2])");

    let either = Filter::eq("a", 1_i64)
        .unwrap()
        .or(Filter::eq("b", 2_i64).unwrap());
    assert_eq!(rendered(either), "(@a:[1 1] | @b:[2 2])");
}

#[test]
fn tag_values_are_escaped() {
    let f = Filter::eq("name", "a-b").unwrap();
    assert_eq!(rendered(f), r"@name:{a\-b}");
}

#[test]
fn field_names_are_escaped_per_segment() {
    let f = Filter::eq("we ird", "x").unwrap();
    assert_eq!(rendered(f), r"@we\ ird:{x}");
}

#[test]
fn redis_number_rejects_non_finite() {
    assert!(RedisNumber::new(f64::NAN).is_err());
    assert!(RedisNumber::new(f64::INFINITY).is_err());
    assert!(RedisNumber::new(f64::NEG_INFINITY).is_err());
    assert!(RedisNumber::new(1.5).is_ok());
}

#[test]
fn core_filter_rejects_unsupported_json() {
    let null = CoreFilter::Eq("k".to_string(), serde_json::Value::Null);
    assert!(Filter::try_from(null).is_err());

    let arr = CoreFilter::Eq("k".to_string(), serde_json::json!([1, 2]));
    assert!(Filter::try_from(arr).is_err());

    let obj = CoreFilter::Eq("k".to_string(), serde_json::json!({"a": 1}));
    assert!(Filter::try_from(obj).is_err());
}

#[test]
fn core_filter_maps_supported_variants() {
    let core = CoreFilter::And(
        Box::new(CoreFilter::Eq(
            "cat".to_string(),
            serde_json::json!("books"),
        )),
        Box::new(CoreFilter::Gt("price".to_string(), serde_json::json!(10))),
    );
    let f = Filter::try_from(core).unwrap();
    assert_eq!(rendered(f), "(@cat:{books} @price:[(10 +inf])");
}

#[test]
fn escape_helpers_are_exported() {
    assert_eq!(escape_tag_value("a|b"), r"a\|b");
    assert_eq!(escape_text_value("a@b"), r"a\@b");
}
