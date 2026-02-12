#!/bin/bash
# ============================================================
# run_test.sh — Dejavu Audio Fingerprint Test Runner
#
# Ye script HOST machine pe run hota hai (Docker ke andar NAHI)
# Run karne ka tarika:
#   cd dejavu
#   bash run_test.sh
#
# Kya karta hai ye script:
#   1. Docker containers start karta hai (PostgreSQL DB + Python)
#   2. Database ready hone ka wait karta hai
#   3. test_my_songs.py run karta hai container ke andar:
#      - 21 original songs fingerprint karta hai (DB mein store)
#      - Agar song pehle se DB mein hai toh SKIP kar deta hai
#      - 21 remix (cover) files ko recognize karta hai
#      - 252 short clips (5-10 sec) ko recognize karta hai
#      - Har test ka PASS/FAIL result dikhata hai
#      - Sab results dataset/results.json mein save karta hai
# ============================================================

set -e  # Koi bhi command fail ho toh script ruk jaye

# ---------- STEP 1: Purane containers + database clear karo ----------
# -v flag: volumes bhi delete karta hai (database reset)
# Har baar fresh start — purane "original" naam wale entries nahi rahenge
echo ""
echo "============================================"
echo "  STEP 1: Clearing old containers + database"
echo "============================================"
docker compose down -v 2>/dev/null || true
echo "  Done. Fresh database will be created."

# ---------- STEP 2: Containers build + start karo ----------
# docker compose up -d: background mein start karta hai
# --build: Dockerfile mein kuch change hua ho toh rebuild kare
# Ye 2 containers start karta hai:
#   1. db      — PostgreSQL database (fingerprints store hote hain)
#   2. python  — Python environment (dejavu code run hota hai)
echo ""
echo "============================================"
echo "  STEP 2: Building & starting containers"
echo "============================================"
docker compose up -d --build
echo "  Containers started."

# ---------- STEP 3: PostgreSQL ready hone ka wait karo ----------
# Database ko start hone mein kuch seconds lagte hain
# pg_isready: PostgreSQL ka built-in health check command
# Jab tak DB ready nahi hota, har 2 sec pe check karta hai
echo ""
echo "============================================"
echo "  STEP 3: Waiting for PostgreSQL to be ready"
echo "============================================"
echo -n "  Waiting"
MAX_RETRIES=30
RETRY=0
until docker compose exec db pg_isready -U postgres -q 2>/dev/null; do
    echo -n "."
    sleep 2
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo ""
        echo "  ERROR: PostgreSQL did not start in 60 seconds!"
        echo "  Check logs: docker compose logs db"
        exit 1
    fi
done
echo ""
echo "  PostgreSQL is ready!"

# ---------- STEP 4: Test script run karo container ke andar ----------
# docker compose run: ek naya Python container create karta hai
# python run_commands/test_my_songs.py: ye script andar run hota hai
#
# Test script 3 kaam karta hai:
#   a) FINGERPRINT: 21 originals ka audio fingerprint DB mein store karo
#      (agar pehle se hai toh skip)
#   b) RECOGNIZE REMIXES: 21 cover songs ko match karo DB se
#   c) RECOGNIZE CLIPS: 252 short clips ko match karo DB se
#   d) Results JSON mein save karo
echo ""
echo "============================================"
echo "  STEP 4: Running test_my_songs.py"
echo "============================================"
echo ""
docker compose run --rm python python run_commands/test_my_songs.py

# ---------- STEP 5: Results file ka location bata do ----------
# Results JSON file dataset/ folder mein save hoti hai
# Volume mount ki wajah se host machine pe bhi accessible hai
echo ""
echo "============================================"
echo "  DONE!"
echo "============================================"
echo ""
echo "  Results saved to: ../dataset/results.json"
echo "  (Full path: $(cd .. && pwd)/dataset/results.json)"
echo ""

# ---------- STEP 6: User se pucho — containers band karne hain? ----------
# Y: containers band ho jayenge (DB data safe rahega)
# N: containers chalu rahenge (manually band karna: docker compose down)
read -p "  Stop containers? (y/n): " STOP_CHOICE
if [ "$STOP_CHOICE" = "y" ] || [ "$STOP_CHOICE" = "Y" ]; then
    echo "  Stopping containers..."
    docker compose down
    echo "  Containers stopped. Database data is safe."
    echo "  (To DELETE database too: docker compose down -v)"
else
    echo "  Containers still running."
    echo "  To stop later: cd dejavu && docker compose down"
fi

echo ""
echo "Done!"
