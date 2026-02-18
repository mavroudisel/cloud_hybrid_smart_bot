#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting Virtual Environment Build..."

# 1. Δημιουργία Virtual Environment (Το "Κουτί")
python3 -m venv venv
source venv/bin/activate

# 2. Εγκατάσταση Βιβλιοθηκών ΜΕΣΑ στο κουτί
pip install --upgrade pip
pip install -r requirements.txt

# 3. Καθαρισμός παλιών αρχείων Stockfish
rm -rf stockfish stockfish.tar temp_sf

# 4. Κατέβασμα Stockfish
echo "📥 Downloading Stockfish..."
curl -L -o stockfish.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar

# 5. Εξαγωγή & Μετακίνηση
mkdir temp_sf
tar -xf stockfish.tar -C temp_sf --strip-components=1
mv temp_sf/stockfish-ubuntu-x86-64-avx2 ./stockfish
chmod +x stockfish

# 6. Καθαρισμός
rm -rf stockfish.tar temp_sf

echo "✅ Build Complete inside VENV!"
