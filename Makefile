# Development tasks for rig-redis-vectorstore.

.PHONY: all check fmt fmt-check clippy test doc integration

all: fmt-check clippy test doc

check:
	cargo check --all-targets

fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test

doc:
	cargo doc --no-deps

# Run the live integration tests. Requires Docker/Podman, or set REDIS_URL to an
# existing RediSearch instance.
integration:
	cargo test --test integration_tests -- --ignored
