#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting Robust Build..."

# 1. Διόρθωση Python Path (Αυτό λύνει το ModuleNotFoundError)
# Χρησιμοποιούμε 'python3 -m pip' για να είμαστε σίγουροι ότι εγκαθιστά
# τις βιβλιοθήκες στον ίδιο python που θα τρέξει μετά.
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 2. Καθαρισμός παλιών αρχείων (για να μην έχουμε conflict)
rm -rf stockfish stockfish.tar temp_sf

# 3. Κατέβασμα Stockfish
echo "📥 Downloading Stockfish..."
curl -L -o stockfish.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar

# 4. Δημιουργία προσωρινού φακέλου και εξαγωγή εκεί
# (Αυτό λύνει το πρόβλημα με το mv error)
mkdir temp_sf
tar -xf stockfish.tar -C temp_sf --strip-components=1

# 5. Μετακίνηση του σωστού αρχείου στο root folder
echo "🔄 Moving binary..."
mv temp_sf/stockfish-ubuntu-x86-64-avx2 ./stockfish

# 6. Δικαιώματα και Καθαρισμός
chmod +x stockfish
rm -rf stockfish.tar temp_sf

echo "✅ Build Complete!"
# Έλεγχος ότι όλα είναι σωστά
ls -l stockfish
python3 -c "import berserk; print('Berserk is installed correctly!')"
