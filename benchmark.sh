#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/verify.sh"

printf '\n==> running Basalt benchmark suite\n'
java --add-modules jdk.incubator.vector -Xmx2g \
  -cp "$ROOT/build/classes:$ROOT/build/test-classes" basalt.tests.BasaltBenchmark "$@" \
  | tee "$ROOT/build/logs/benchmark.log"

PYTHON_BIN=""
for candidate in "${PYTHON:-}" python3 python \
    "/c/Users/Coffee/AppData/Local/Programs/Python/Python311/python.exe"; do
  if [ -n "$candidate" ] && "$candidate" -c "import matplotlib" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -n "$PYTHON_BIN" ]; then
  printf '\n==> generating report figures with %s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" "$ROOT/report/make_figures.py"
  printf '\n==> rendering self-contained HTML report\n'
  "$PYTHON_BIN" "$ROOT/report/make_report.py"
else
  printf '\nmatplotlib is unavailable; CSV files remain in build/bench and figures were not regenerated.\n'
fi
