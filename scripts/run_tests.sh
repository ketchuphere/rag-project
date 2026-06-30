#!/bin/bash
# Run all unit and integration tests.
set -e

echo "Running unit tests..."
python -m pytest tests/unit/ -v

echo ""
echo "Running integration tests..."
python -m pytest tests/integration/ -v

echo ""
echo "✅ All tests passed."
