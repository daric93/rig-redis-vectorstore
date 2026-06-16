# Contributing to rig-redis-vectorstore

Contributions are welcome and appreciated. This is a community-maintained Redis
(RediSearch) vector store integration for the [Rig](https://crates.io/crates/rig-core)
framework, and improvements, bug reports, and ideas from anyone are encouraged.

## Open an issue before a pull request

Please **open an issue first** and wait for a brief discussion before sending a
pull request, so we can agree on the approach and avoid duplicated or wasted work.

- **Bugs:** include the crate version, Redis/RediSearch version (or image, e.g.
  `redis/redis-stack-server:latest`), a minimal reproduction, and what you expected.
- **Features / changes:** describe the use case and proposed API before implementing.

Small, obvious fixes (typos, doc tweaks) may skip the issue — use your judgement.

## Developer Certificate of Origin (sign-off)

All commits must be signed off under the [Developer Certificate of Origin](https://developercertificate.org/).
By signing off you certify that you wrote the patch or otherwise have the right to
submit it under the project's license.

Add the sign-off line automatically with:

```bash
git commit -s -m "your message"
```

This appends `Signed-off-by: Your Name <your@email>` to the commit message (which
must match your git `user.name` / `user.email`). PRs with unsigned commits will be
asked to amend.

## Licensing

This project is licensed under the [MIT License](./LICENSE). By contributing, you
agree that your contributions are licensed under the same MIT terms. Do not submit
code you do not have the right to license under MIT.

## Development setup

```bash
# Build
cargo build

# Format (CI enforces --check)
cargo fmt

# Lint (CI denies warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Unit + filter tests (no services needed)
cargo test

# Docs
cargo doc --no-deps
```

A `Makefile` wraps these: `make fmt-check`, `make clippy`, `make test`, `make doc`,
or `make all`.

### Integration tests

Integration tests are marked `#[ignore]` and require a Redis instance with the
RediSearch module. Start one and run them explicitly:

```bash
# Start Redis Stack (Docker or Podman)
podman run -d --name redis -p 6379:6379 redis/redis-stack-server:latest

# Run the ignored integration tests against it
REDIS_URL=redis://127.0.0.1:6379 cargo test --test integration_tests -- --ignored
```

Without `REDIS_URL`, the tests try to start a container via testcontainers and skip
gracefully if neither Docker/Podman nor a reachable RediSearch instance is available.

## Pull request checklist

Before requesting review, please make sure:

- [ ] There is a linked issue describing the change.
- [ ] `cargo fmt --check` passes.
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` is clean.
- [ ] `cargo test` passes, and integration tests pass against a live RediSearch
      instance when your change affects Redis behavior.
- [ ] Public items have `///` docs and the `CHANGELOG.md` is updated.
- [ ] All commits are signed off (`git commit -s`).

## Code style

- Follow the existing module layout and naming.
- Use the `rig-core` error types (`VectorStoreError`, `FilterError`) — not `String`
  errors — for fallible public APIs.
- Keep changes scoped; avoid unrelated refactors in the same PR.
- Add or update tests for the behavior you change.

Thanks for contributing!
