# syntax=docker/dockerfile:1.7

FROM rust:1.85-bookworm AS builder
WORKDIR /build

# Cache dependencies first
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

RUN cargo build --release --bin warlock

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/warlock /usr/local/bin/warlock
COPY warlock.docker.toml /app/warlock.docker.toml

ENTRYPOINT ["warlock"]
CMD ["--help"]
