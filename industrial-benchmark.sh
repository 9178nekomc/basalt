#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/verify.sh"
exec java --add-modules jdk.incubator.vector \
  -cp "$ROOT/build/classes:$ROOT/build/test-classes" basalt.tests.IndustrialBenchmark "$@"
