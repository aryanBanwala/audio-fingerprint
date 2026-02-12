#!/bin/bash
# Step 2: Fingerprint (load) songs into database
# Run this INSIDE the Docker container

# ---- CHANGE THESE AS NEEDED ----
SONG_FOLDER="mp3"        # folder containing your audio files
EXTENSION="mp3"           # file extension (mp3, wav, etc.)
# ---------------------------------

echo "Fingerprinting all .$EXTENSION files in $SONG_FOLDER/ ..."
python dejavu.py -c dejavu_docker.cnf --fingerprint "$SONG_FOLDER" "$EXTENSION"
echo ""
echo "Done! Songs are now in the database."
echo "You can now recognize audio using: python dejavu.py --recognize file <path_to_audio>"
