#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CLASSES="$ROOT/build/classes"
TEST_CLASSES="$ROOT/build/test-classes"
FAIL_CLASSES="$ROOT/build/compile-fail-classes"
LOG_DIR="$ROOT/build/logs"
mkdir -p "$CLASSES" "$TEST_CLASSES" "$FAIL_CLASSES" "$LOG_DIR"

# Build and unit-test the Rust core before compiling Java. Set JSL_SKIP_NATIVE=1 for a pure-Java
# verification run on a machine without a Rust toolchain.
if [ "${JSL_SKIP_NATIVE:-0}" != "1" ] && command -v cargo >/dev/null 2>&1; then
  printf '==> building and testing native kernels\n'
  "$ROOT/native-build.sh"
else
  printf '==> native kernels skipped (JSL_SKIP_NATIVE=%s, cargo=%s)\n' \
    "${JSL_SKIP_NATIVE:-0}" "$(command -v cargo || printf missing)"
fi

JAVAC_FLAGS=(--release 21 --add-modules jdk.incubator.vector -Xlint:all
  -J-Duser.language=en -J-Duser.country=US)

# The incubating-module notice is emitted by javac itself and cannot be suppressed with -Xlint.
# Every other warning is treated as an error.
check_warnings() {
  local log="$1"
  local total incubating
  total=$(grep -c "warning:" "$log" || true)
  incubating=$(grep -c "using incubating module" "$log" || true)
  if [ "$total" -ne "$incubating" ]; then
    printf 'Unexpected compiler warnings:\n' >&2
    cat "$log" >&2
    exit 1
  fi
}

printf '==> compiling main sources (-Xlint:all)\n'
javac "${JAVAC_FLAGS[@]}" -d "$CLASSES" \
  "$ROOT/src/main/java/jsl/"*.java \
  "$ROOT/src/main/java/jsl/core/"*.java \
  "$ROOT/src/main/java/jsl/io/"*.java \
  "$ROOT/src/main/java/jsl/ml/"*.java \
  "$ROOT/src/main/java/jsl/nn/"*.java \
  "$ROOT/src/main/java/jsl/data/"*.java \
  "$ROOT/src/main/java/jsl/concurrent/"*.java \
  "$ROOT/src/main/java/jsl/nativebridge/"*.java \
  "$ROOT/src/main/java/jsl/torch/"*.java > "$LOG_DIR/main-javac.log" 2>&1
check_warnings "$LOG_DIR/main-javac.log"

printf '==> compiling test sources (-Xlint:all)\n'
mapfile -t TEST_SOURCES < <(find "$ROOT/src/test/java" -name "*.java" \
  ! -name "JslApiTest.java" ! -name "PerformanceTest.java" -print)
javac "${JAVAC_FLAGS[@]}" -cp "$CLASSES" -d "$TEST_CLASSES" \
  "${TEST_SOURCES[@]}" > "$LOG_DIR/test-javac.log" 2>&1
check_warnings "$LOG_DIR/test-javac.log"

printf '==> running classical suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.TestSuite
printf '==> running industrial ML suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.IndustrialTestSuite
printf '==> running sequence suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.SequenceTestSuite
printf '==> running breadth suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.BreadthTestSuite
printf '==> running native tensor suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.NativeTestSuite
printf '==> running torch eager/autograd suite\n'
java --add-modules jdk.incubator.vector -cp "$CLASSES:$TEST_CLASSES" jsl.tests.TorchTestSuite

if [ -f "$ROOT/pom.xml" ] && command -v mvn >/dev/null 2>&1; then
  printf '==> running Maven JUnit 5 SDK tests\n'
  mvn -q -f "$ROOT/pom.xml" test
else
  printf '==> Maven not found; JUnit 5 SDK tests skipped\n'
fi

printf '==> verifying compile-time rejection\n'
for source in "$ROOT/src/test/compile-fail/jsl/tests/"*.java; do
  if javac "${JAVAC_FLAGS[@]}" -cp "$CLASSES" -d "$FAIL_CLASSES" "$source" \
      > "$FAIL_CLASSES/diagnostic.log" 2>&1; then
    printf 'Compile-time safety failure: %s unexpectedly compiled\n' "$source" >&2
    exit 1
  fi
  printf 'OK: rejected at compile time: %s\n' "$(basename "$source")"
done

printf 'All JSL verification checks passed.\n'
