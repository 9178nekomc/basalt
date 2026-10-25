#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/build/native"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) CPU="aarch64" ;;
  *) CPU="x86_64" ;;
esac
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM="windows-$CPU"
    LIB_NAME="basalt_native.dll"
    ;;
  Darwin*)
    PLATFORM="macos-$CPU"
    LIB_NAME="libbasalt_native.dylib"
    ;;
  *)
    PLATFORM="linux-$CPU"
    LIB_NAME="libbasalt_native.so"
    ;;
esac

mkdir -p "$ROOT/src/main/resources/native/$PLATFORM"

if command -v cargo >/dev/null 2>&1; then
  printf '==> building and testing Rust native kernels via cargo
'
  cd "$ROOT/native"
  cargo test --release
  cargo build --release
  cp "target/release/$LIB_NAME" "$ROOT/build/native/$LIB_NAME"
  cp "target/release/$LIB_NAME" "$ROOT/src/main/resources/native/$PLATFORM/$LIB_NAME"
elif command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
  CXX_BIN="$(command -v g++ || command -v clang++)"
  printf '==> cargo not found; compiling C++ JNI native kernels via %s
' "$CXX_BIN"
  "$CXX_BIN" -O3 -std=c++17 -fPIC -shared -pthread -I "$ROOT/native/cpp"     "$ROOT/native/cpp/basalt_native.cpp" -o "$ROOT/build/native/$LIB_NAME"
  cp "$ROOT/build/native/$LIB_NAME" "$ROOT/src/main/resources/native/$PLATFORM/$LIB_NAME"
else
  printf '==> no native compiler found; using prebuilt native binary if available
'
fi

printf 'native library ready: %s (packaged for %s)
' "$ROOT/build/native/$LIB_NAME" "$PLATFORM"
