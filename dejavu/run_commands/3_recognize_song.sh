#!/bin/bash
# Step 3: Recognize (match) an audio file
# Run this INSIDE the Docker container

if [ -z "$1" ]; then
    echo "Usage: bash 3_recognize_song.sh <path_to_audio_file>"
    echo "Example: bash 3_recognize_song.sh mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3"
    exit 1
fi

echo "Recognizing: $1"
python dejavu.py -c dejavu_docker.cnf --recognize file "$1"
