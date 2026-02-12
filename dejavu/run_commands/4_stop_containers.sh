#!/bin/bash
# Step 4: Stop everything
# Run this from the dejavu/ root folder (NOT inside container)

echo "Stopping Dejavu containers..."
docker compose down
echo "Done! Containers stopped."
echo ""
echo "Note: Your fingerprints are safe. They'll be there when you start again."
echo "To DELETE everything (including database): docker compose down -v"
