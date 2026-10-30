#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN=""
for candidate in "${PYTHON:-}" python3 python \
    "/c/Users/Coffee/AppData/Local/Programs/Python/Python311/python.exe"; do
  if [ -n "$candidate" ] && "$candidate" -c "import sklearn, matplotlib, numpy" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  printf 'This comparison requires Python with scikit-learn, numpy and matplotlib.\n' >&2
  exit 1
fi

printf '==> verifying Basalt\n'
"$ROOT/verify.sh"

printf '\n==> generating shared data and running scikit-learn\n'
"$PYTHON_BIN" "$ROOT/report/compare_reference.py"

printf '\n==> running Basalt on the same data\n'
java --add-modules jdk.incubator.vector -Xmx3g \
  -cp "$ROOT/build/classes:$ROOT/build/test-classes" basalt.tests.ReferenceComparison

printf '\n==> running concurrency benchmarks\n'
java --add-modules jdk.incubator.vector -Xmx2g \
  -cp "$ROOT/build/classes:$ROOT/build/test-classes" basalt.tests.ConcurrencyBenchmark
"$PYTHON_BIN" "$ROOT/report/compare_concurrency.py"

printf '\n==> rendering comparison figures and tables\n'
"$PYTHON_BIN" "$ROOT/report/compare_plots.py"

printf '\n==> rendering self-contained HTML\n'
"$PYTHON_BIN" "$ROOT/report/make_report.py" "$ROOT/report/Basalt_VS_REFERENCE.md"

printf '\nComparison complete: report/Basalt_VS_REFERENCE.html\n'
