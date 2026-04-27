#!/bin/bash
# scripts/olaf_test.sh
# Run from repo root:
#   bash scripts/olaf_test.sh                  # default: dataset/
#   bash scripts/olaf_test.sh --data-dir music # override

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/dataset"
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# Resolve relative
case "$DATA_DIR" in
    /*) ;;
    *) DATA_DIR="$ROOT/$DATA_DIR" ;;
esac

echo ""
echo "=========================================="
echo "  OLAF TEST  (using music/)"
echo "=========================================="

# Sanity: required tools
for tool in ffmpeg gcc make python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found. Install it first."
        exit 1
    fi
done

# Build olaf if needed
if [ ! -x "$ROOT/olaf/bin/olaf_c" ]; then
    echo "[build] olaf binary missing — running 'make' in olaf/"
    (cd "$ROOT/olaf" && make)
fi

python3 "$ROOT/scripts/olaf_test.py" --data-dir "$DATA_DIR"

DATA_TAG=$(basename "$DATA_DIR")
echo ""
echo "Done. Results: $ROOT/scripts/results/olaf_${DATA_TAG}_results.json"
