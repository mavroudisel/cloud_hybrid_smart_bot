#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

echo "📥 Downloading Stockfish..."
curl -L -o stockfish.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar
tar -xf stockfish.tar
mv stockfish/stockfish-ubuntu-x86-64-avx2 ./stockfish
chmod +x stockfish
rm -rf stockfish.tar stockfish/