#!/usr/bin/env bash
# Cross-platform downloader implementation lives beside this wrapper.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/download_stackoverflow.py" "$@"
