#!/bin/bash
# Step 1: Start Docker containers
# Run this from the dejavu/ root folder

echo "Starting Dejavu containers..."
docker compose up -d

echo ""
echo "Containers started! Now run:"
echo "  docker compose run python /bin/bash"
echo ""
echo "This will give you a shell inside the container."
echo "Then use the fingerprint and recognize commands."
