#!/bin/bash
# scripts/dejavu_test.sh
# Run from repo root:
#   bash scripts/dejavu_test.sh                  # default: dataset/
#   bash scripts/dejavu_test.sh --data-dir music # override
#
# Spins up dejavu's docker stack (postgres + python), wipes DB, then runs
# /code/scripts/dejavu_test.py inside the python container.

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEJAVU_DIR="$ROOT/dejavu"

DATA_DIR="dataset"
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo ""
echo "=========================================="
echo "  DEJAVU TEST  (using music/)"
echo "=========================================="

cd "$DEJAVU_DIR"

echo ""
echo "[1/4] Wiping previous DB + containers..."
docker compose down -v 2>/dev/null || true

echo ""
echo "[2/4] Building & starting containers..."
docker compose up -d --build

echo ""
echo "[3/4] Waiting for PostgreSQL..."
RETRY=0
until docker compose exec -T db pg_isready -U postgres -q 2>/dev/null; do
    printf "."
    sleep 2
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge 30 ]; then
        echo " ERROR: PostgreSQL did not start in 60s"
        docker compose logs db | tail -30
        exit 1
    fi
done
echo " ready."

echo ""
echo "[4/4] Running scripts/dejavu_test.py inside python container (data: $DATA_DIR)..."
docker compose run --rm python python /code/scripts/dejavu_test.py --data-dir "$DATA_DIR"

echo ""
echo "Done. Results: $ROOT/scripts/results/dejavu_${DATA_DIR}_results.json"

read -p "Stop containers? (y/n): " STOP
if [[ "$STOP" =~ ^[Yy]$ ]]; then
    docker compose down
    echo "Containers stopped."
else
    echo "Containers still running."
fi
