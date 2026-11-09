#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/download_ngram_table.py" "$@"
