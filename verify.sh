#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Everything below uses paths relative to ROOT: an absolute path containing a
# space breaks both @argfile contents and direct javac arguments on Windows.
cd "$ROOT"
CLASSES="build/classes"
TEST_CLASSES="build/test-classes"
FAIL_CLASSES="build/compile-fail-classes"
LOG_DIR="build/logs"
mkdir -p "$CLASSES" "$TEST_CLASSES" "$FAIL_CLASSES" "$LOG_DIR"

# java.exe on Windows wants ';' between classpath entries; POSIX shells use ':'.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) CP_SEP=';' ;;
  *) CP_SEP=':' ;;
esac
TEST_CP="$CLASSES$CP_SEP$TEST_CLASSES"

if [ "${BASALT_SKIP_NATIVE:-0}" != "1" ]; then
  printf '==> building and testing native kernels
'
  bash ./native-build.sh
fi

if command -v javac >/dev/null 2>&1; then
  JAVAC_CMD=(javac)
else
  JAVAC_CMD=(java -m jdk.compiler/com.sun.tools.javac.Main)
fi

JAVA_MAJOR=$(java -version 2>&1 | head -n 1 | sed -E 's/.*"([0-9]+).*/\1/')
if [ "$JAVA_MAJOR" -ge 21 ] 2>/dev/null; then
  RELEASE_VER=21
else
  RELEASE_VER=17
fi

# -J-Duser.language forces English diagnostics: check_warnings greps for the
# literal "warning:", which never matches a localized javac.
JAVAC_FLAGS=(-J-Duser.language=en -J-Duser.country=US -encoding UTF-8 \
  -sourcepath src/main/java --release "$RELEASE_VER" \
  --add-modules jdk.incubator.vector -Xlint:all)

# javac reads @argfiles verbatim (no MSYS translation), so the file must contain
# paths it can resolve; relative forward-slash paths work on every platform.
emit_sources() {
  local dir="$1" out="$2"
  shift 2
  find "$dir" -name "*.java" "$@" > "$out"
}

check_warnings() {
  local log="$1"
  local total incubating
  total=$(grep -c "warning:" "$log" || true)
  incubating=$(grep -c "using incubating module" "$log" || true)
  if [ "$total" -ne "$incubating" ]; then
    printf 'Unexpected compiler warnings:
' >&2
    cat "$log" >&2
    exit 1
  fi
}

printf '==> compiling main sources (-Xlint:all, release %s)
' "$RELEASE_VER"
emit_sources src/main/java "$LOG_DIR/main-sources.txt"
"${JAVAC_CMD[@]}" "${JAVAC_FLAGS[@]}" -d "$CLASSES" @"$LOG_DIR/main-sources.txt" > "$LOG_DIR/main-javac.log" 2>&1
check_warnings "$LOG_DIR/main-javac.log"

printf '==> compiling test sources (-Xlint:all)
'
emit_sources src/test/java "$LOG_DIR/test-sources.txt" \
  ! -name "BasaltApiTest.java" ! -name "PerformanceTest.java" \
  ! -name "AdvancedApiTest.java"
"${JAVAC_CMD[@]}" "${JAVAC_FLAGS[@]}" -cp "$CLASSES" -d "$TEST_CLASSES" @"$LOG_DIR/test-sources.txt" > "$LOG_DIR/test-javac.log" 2>&1
check_warnings "$LOG_DIR/test-javac.log"

printf '==> running classical suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.TestSuite
printf '==> running industrial ML suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.IndustrialTestSuite
printf '==> running sequence suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.SequenceTestSuite
printf '==> running breadth suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.BreadthTestSuite
printf '==> running native tensor suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.NativeTestSuite
printf '==> running torch eager/autograd suite
'
java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.TorchTestSuite

if [ -f "$TEST_CLASSES/basalt/tests/Phase1TestSuite.class" ]; then
  printf '==> running Phase 1 suite (JNI engineering, preprocessing, model selection, pipeline)
'
  java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.Phase1TestSuite
fi

if [ -f "$TEST_CLASSES/basalt/tests/Phase2TestSuite.class" ]; then
  printf '==> running Phase 2 suite (kernels, unsupervised, feature eng, DAG, LR schedulers, ONNX)
'
  java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.Phase2TestSuite
fi

if [ -f "$TEST_CLASSES/basalt/tests/Phase3TestSuite.class" ]; then
  printf '==> running Phase 3 suite (explainability SHAP/LIME, JFR & observability metrics)
'
  java --add-modules jdk.incubator.vector -cp "$TEST_CP" basalt.tests.Phase3TestSuite
fi

if [ -f "$ROOT/pom.xml" ] && command -v mvn >/dev/null 2>&1; then
  printf '==> running Maven JUnit 5 SDK tests
'
  mvn -q test
else
  printf '==> Maven not found; JUnit 5 SDK tests skipped
'
fi

printf '==> verifying compile-time rejection
'
for source in src/test/compile-fail/basalt/tests/*.java; do
  if "${JAVAC_CMD[@]}" "${JAVAC_FLAGS[@]}" -cp "$CLASSES" -d "$FAIL_CLASSES" "$source"       > "$FAIL_CLASSES/diagnostic.log" 2>&1; then
    printf 'Compile-time safety failure: %s unexpectedly compiled
' "$source" >&2
    exit 1
  fi
  printf 'OK: rejected at compile time: %s
' "$(basename "$source")"
done

printf 'All Basalt verification checks passed.
'
