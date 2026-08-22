#!/usr/bin/env bash

run_test() {
  local name="$1"
  shift
  echo "# Start: $name"
  if "$@"; then
    echo "# End: Pass $name"
    return 0
  else
    echo "# End: Fail $name"
    return 1
  fi
}

echo "# SD.Next Static Checks"
echo "# Activate VENV"
source venv/bin/activate || true
echo "# Install toolkit"
pip install -U pre-commit ruff pylint codespell ty pyright || true

run_test "Precommit" npm run precommit || true
run_test "ESLint" npm run eslint || true
run_test "TypeScript" npm run tsc || true
run_test "Ruff" npm run ruff || true
run_test "Pylint" npm run pylint || true
run_test "Codespell" npm run codespell || true
run_test "Ty" npm run ty || true
run_test "Pyright" npm run pyright || true

echo "Done..."
