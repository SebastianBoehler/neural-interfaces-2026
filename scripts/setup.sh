#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --recursive
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pip \
  -r requirements/baseline-macos-arm64.txt
