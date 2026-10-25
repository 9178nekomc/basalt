#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/native"

# Unit tests cover SIMD tails, broadcast gradients, graph ordering, K-Means and the classic
# fast paths. They run before the release build so a broken kernel never reaches build/native.
cargo test --release

cargo build --release

mkdir -p "$ROOT/build/native"
ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) CPU="aarch64" ;;
  *) CPU="x86_64" ;;
esac
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM="windows-$CPU"
    cp target/release/jsl_native.dll "$ROOT/build/native/jsl_native.dll"
    mkdir -p "$ROOT/src/main/resources/native/$PLATFORM"
    cp target/release/jsl_native.dll "$ROOT/src/main/resources/native/$PLATFORM/jsl_native.dll"
    ;;
  Darwin*)
    PLATFORM="macos-$CPU"
    cp target/release/libjsl_native.dylib "$ROOT/build/native/libjsl_native.dylib"
    mkdir -p "$ROOT/src/main/resources/native/$PLATFORM"
    cp target/release/libjsl_native.dylib "$ROOT/src/main/resources/native/$PLATFORM/libjsl_native.dylib"
    ;;
  *)
    PLATFORM="linux-$CPU"
    cp target/release/libjsl_native.so "$ROOT/build/native/libjsl_native.so"
    mkdir -p "$ROOT/src/main/resources/native/$PLATFORM"
    cp target/release/libjsl_native.so "$ROOT/src/main/resources/native/$PLATFORM/libjsl_native.so"
    ;;
esac

printf 'native library ready: %s (packaged for %s)\n' "$ROOT/build/native" "$PLATFORM"
