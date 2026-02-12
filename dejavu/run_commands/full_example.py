"""
Full example: Fingerprint songs + Recognize audio
Run this INSIDE the Docker container: python run_commands/full_example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

# Database config (matches docker-compose.yaml)
config = {
    "database": {
        "host": "db",
        "user": "postgres",
        "password": "password",
        "database": "dejavu"
    },
    "database_type": "postgres"
}

if __name__ == '__main__':
    # Connect to database
    djv = Dejavu(config)

    # ========== STEP 1: FINGERPRINT ==========
    # Load all mp3 files from mp3/ folder into database
    print("=" * 50)
    print("STEP 1: Fingerprinting songs...")
    print("=" * 50)
    djv.fingerprint_directory("mp3", [".mp3"])
    print(f"Total fingerprints in database: {djv.db.get_num_fingerprints()}")
    print(f"Total songs in database: {djv.db.get_num_songs()}")
    print()

    # ========== STEP 2: RECOGNIZE ==========
    # Try to recognize a song from file
    print("=" * 50)
    print("STEP 2: Recognizing a song...")
    print("=" * 50)
    results = djv.recognize(FileRecognizer, "mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
    print(f"Result: {results}")
    print()

    # Show the best match
    if results and results.get("results"):
        best = results["results"][0]
        print(f"Best match: {best['song_name']}")
        print(f"Confidence: {best['input_confidence']}")
        print(f"Total time: {results['total_time']:.2f}s")
