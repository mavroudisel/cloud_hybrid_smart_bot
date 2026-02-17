#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting Build Process..."

# 1. Install Dependencies
pip install -r requirements.txt

# 2. Clean up previous attempts (για να μην μπερδεύεται το mv)
rm -rf stockfish stockfish_folder stockfish.tar

# 3. Download Stockfish
echo "📥 Downloading Stockfish..."
curl -L -o stockfish.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar

# 4. Extract
echo "📂 Extracting..."
tar -xf stockfish.tar

# 5. Find the binary and move it
# Αυτή η εντολή βρίσκει το αρχείο όπου κι αν είναι και το φέρνει εδώ με το όνομα 'stockfish'
echo "🔍 Locating and moving binary..."
find . -name "stockfish-ubuntu-x86-64-avx2" -type f -exec mv {} ./stockfish \;

# 6. Make executable
chmod +x stockfish

# 7. Cleanup
rm stockfish.tar
# Σβήνουμε τυχόν φακέλους που έμειναν
find . -type d -name "stockfish-*" -exec rm -rf {} +

echo "✅ Build Complete! Stockfish is ready."
ls -l stockfish
