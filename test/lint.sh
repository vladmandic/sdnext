#!/usr/bin/env bash

run_test() {
  local name="$1"
  shift
  local start=$SECONDS
  echo "# Start: $name"
  if "$@"; then
    local elapsed=$((SECONDS - start))
    echo "# Pass: $name time=${elapsed}"
    return 0
  else
    local status=$?
    local elapsed=$((SECONDS - start))
    echo "# Fail: $name time=${elapsed} exit=${status}"
    return "$status"
  fi
}

echo "# SD.Next Static Checks"
run_test "venv" source venv/bin/activate || true
run_test "install" pip install -U pre-commit ruff pylint codespell ty pyright || true

run_test "lint:precommit" npm run lint:precomit || true
run_test "lint:ui" npm run lint:ui || true
run_test "tsc:ui" npm run tsc:ui || true
run_test "lint:ruff" npm run lint:ruff || true
run_test "lint:pylint" npm run lint:pylint || true
run_test "lint:codespell" npm run lint:codespell || true
run_test "lint:ty" npm run lint:ty || true
run_test "lint:md" npm run lint:md || true
run_test "lint:pyright" npm run lint:pyright || true

echo "Done..."
