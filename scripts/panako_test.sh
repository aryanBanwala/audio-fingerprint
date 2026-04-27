#!/bin/bash
# scripts/panako_test.sh
# Run from repo root:
#   bash scripts/panako_test.sh                  # default: dataset/
#   bash scripts/panako_test.sh --data-dir music # override

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
case "$DATA_DIR" in
    /*) ;;
    *) DATA_DIR="$ROOT/$DATA_DIR" ;;
esac

echo ""
echo "=========================================="
echo "  PANAKO TEST  (using music/)"
echo "=========================================="

for tool in ffmpeg java python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found. Install it first."
        exit 1
    fi
done

# Build jar if needed
if ! ls "$ROOT"/Panako/build/libs/*-all.jar >/dev/null 2>&1; then
    echo "[build] panako jar missing — running ./gradlew shadowJar"
    (cd "$ROOT/Panako" && ./gradlew shadowJar)
fi

python3 "$ROOT/scripts/panako_test.py" --data-dir "$DATA_DIR"

DATA_TAG=$(basename "$DATA_DIR")
echo ""
echo "Done. Results: $ROOT/scripts/results/panako_${DATA_TAG}_results.json"
