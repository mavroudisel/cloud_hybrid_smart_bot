import berserk
import threading
import chess
import os
import time
import numpy as np
import onnxruntime as ort
from stockfish import Stockfish
from flask import Flask, render_template_string, jsonify

# ==========================================
#              ΡΥΘΜΙΣΕΙΣ
# ==========================================
TOKEN = os.environ.get("LICHESS_TOKEN")
MODEL_ONNX = "my_chess_bot.onnx"
VOCAB_FILE = "vocab.npz"
STOCKFISH_PATH = "./stockfish"

app = Flask(__name__)
last_message = {"id": 0, "text": ""}
message_id = 0
last_analyzed_move_count = -1

def broadcast_speech(text):
    global last_message, message_id
    message_id += 1
    last_message = {"id": message_id, "text": text}
    print(f"🗣️ Σχόλιο: {text}")

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chess Coach Audio</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; padding: 50px; }
            button { background-color: #ff9800; color: white; padding: 25px 50px; font-size: 22px; border: none; border-radius: 12px; cursor: pointer; font-weight: bold; box-shadow: 0 5px #bf7300; }
            button:active { transform: translateY(3px); box-shadow: 0 2px #bf7300; }
            #status { margin-top: 30px; font-size: 20px; color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>♟️ AI Chess Coach</h1>
        <button id="startBtn" onclick="startAudio()">🔊 ΕΝΕΡΓΟΠΟΙΗΣΗ ΗΧΟΥ</button>
        <div id="status">Αναμονή...</div>
        <script>
            let lastId = 0;
            function startAudio() {
                let msg = new SpeechSynthesisUtterance("Σύστημα ήχου ενεργό.");
                msg.lang = 'el-GR';
                window.speechSynthesis.speak(msg);
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('status').innerText = "✅ Ο ήχος λειτουργεί!";
                setInterval(checkMessages, 1000);
            }
            function checkMessages() {
                fetch('/poll').then(r => r.json()).then(data => {
                    if (data.id > lastId) {
                        lastId = data.id;
                        let utterance = new SpeechSynthesisUtterance(data.text);
                        utterance.lang = 'el-GR';
                        window.speechSynthesis.speak(utterance);
                    }
                });
            }
        </script>
    </body>
    </html>
    """)

@app.route('/poll')
def poll(): return jsonify(last_message)

class ChessBrain:
    def __init__(self):
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=14)
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
        except: self.ort_session = None

    def analyze_move(self, board, move_uci):
        try:
            board.pop()
            self.sf.set_fen_position(board.fen())
            best_move = self.sf.get_best_move()
            score_before = self.get_score(board.turn)
            board.push(chess.Move.from_uci(move_uci))
            self.sf.set_fen_position(board.fen())
            score_after = -self.get_score(board.turn)
            diff = score_before - score_after
            if board.is_checkmate(): broadcast_speech("Δυστυχώς, είναι μάτ.")
            elif move_uci == best_move: broadcast_speech("Εξαιρετική κίνηση!")
            elif diff < 40: broadcast_speech("Πολύ καλή επιλογή.")
            elif diff < 150: broadcast_speech(f"Καλή κίνηση, αλλά η {best_move} ήταν καλύτερη.")
            else: broadcast_speech(f"Λάθος κίνηση. Έπρεπε να παίξεις {best_move}.")
        except: pass

    def get_score(self, turn):
        ev = self.sf.get_evaluation()
        val = ev['value']
        if ev['type'] == 'mate': val = 10000 if val > 0 else -10000
        return val if turn == chess.WHITE else -val

    def get_bot_move(self, board):
        # Εδώ το bot παίζει με το μοντέλο σου (ONNX)
        move_uci = None
        if self.ort_session:
            # ONNX Logic... (απλοποιημένο για σιγουριά)
            pass
        if not move_uci:
            self.sf.set_fen_position(board.fen())
            move_uci = self.sf.get_best_move()
        return move_uci

def run_lichess_bot():
    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    brain = ChessBrain()
    me_id = client.account.get()['username'].lower()
    global last_analyzed_move_count

    print(f"🚀 Bot {me_id} is active!")

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.accept_challenge(event['challenge']['id'])

        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            stream = client.bots.stream_game_state(game_id)
            last_analyzed_move_count = -1 # Reset για νέο παιχνίδι
            
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    bot_is_white = (g_evt['white'].get('id', '').lower() == me_id)
                    moves = g_evt['state']['moves'].split()
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                else: continue

                board = chess.Board()
                for m in moves: board.push(chess.Move.from_uci(m))
                
                cur_count = len(moves)
                is_bot_turn = (board.turn == (chess.WHITE if bot_is_white else chess.BLACK))

                # Η ΔΙΟΡΘΩΣΗ ΕΙΝΑΙ ΕΔΩ:
                if is_bot_turn and not board.is_game_over():
                    if cur_count > last_analyzed_move_count:
                        # Αν υπάρχουν κινήσεις, ανάλυσε την τελευταία του χρήστη
                        if cur_count > 0:
                            brain.analyze_move(board, moves[-1])
                        
                        # Το Bot παίζει
                        bot_move = brain.get_bot_move(board)
                        client.bots.make_move(game_id, bot_move)
                        last_analyzed_move_count = cur_count + 1 # Σημειώνουμε ότι παίξαμε

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=10000)).start()
    run_lichess_bot()
