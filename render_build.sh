#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting Build Process..."

# 1. Install Dependencies
pip install -r requirements.txt

# 2. Clean up
rm -f stockfish stockfish.tar

# 3. Download Stockfish
echo "📥 Downloading Stockfish..."
curl -L -o stockfish.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar

# 4. Extract
echo "📂 Extracting..."
tar -xf stockfish.tar

# 5. Rename binary
# Βάσει των logs σου, το αρχείο βγαίνει χύμα με το μακρύ όνομα. Το μετονομάζουμε απλά.
if [ -f "stockfish-ubuntu-x86-64-avx2" ]; then
    mv stockfish-ubuntu-x86-64-avx2 stockfish
    echo "✅ Renamed binary to 'stockfish'"
else
    # Fallback: Αν αλλάξει κάτι και είναι μέσα σε φάκελο
    find . -name "stockfish-ubuntu-x86-64-avx2" -type f -exec mv {} ./stockfish \;
fi

# 6. Make executable
chmod +x stockfish

# 7. Cleanup
rm stockfish.tar

echo "✅ Build Complete! Ready to check file:"
ls -l stockfish
