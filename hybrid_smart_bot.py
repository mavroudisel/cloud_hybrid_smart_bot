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

# ==========================================
#          SITE ΓΙΑ ΤΟΝ ΗΧΟ (ΠΛΗΡΕΣ)
# ==========================================
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
            .info { color: #888; margin-top: 20px; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>♟️ AI Chess Coach</h1>
        <p>Πάτα το κουμπί για να ξεκινήσει ο ήχος</p>
        <button id="startBtn" onclick="startAudio()">🔊 ΕΝΕΡΓΟΠΟΙΗΣΗ ΗΧΟΥ</button>
        <div id="status">Αναμονή για σύνδεση...</div>
        <div class="info">Κράτα αυτή τη σελίδα ανοιχτή κατά τη διάρκεια του παιχνιδιού.</div>

        <script>
            let lastId = 0;
            function startAudio() {
                let msg = new SpeechSynthesisUtterance("Το σύστημα ήχου είναι έτοιμο.");
                msg.lang = 'el-GR';
                window.speechSynthesis.speak(msg);
                
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('status').innerText = "✅ Ο ήχος είναι ενεργός!";
                
                setInterval(checkMessages, 1000);
            }

            function checkMessages() {
                fetch('/poll')
                .then(r => r.json())
                .then(data => {
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

# ==========================================
#          ΕΓΚΕΦΑΛΟΣ ΚΑΙ ΑΝΑΛΥΣΗ
# ==========================================
class ChessBrain:
    def __init__(self):
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=14)
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
            print("✅ ONNX Model Loaded")
        except: self.ort_session = None

    def analyze_move(self, board, move_uci):
        try:
            # Πηγαίνουμε πίσω για να δούμε τι έκανε ο χρήστης
            board.pop()
            self.sf.set_fen_position(board.fen())
            best_move = self.sf.get_best_move()
            score_before = self.get_score(board.turn)
            
            board.push(chess.Move.from_uci(move_uci))
            self.sf.set_fen_position(board.fen())
            score_after = -self.get_score(board.turn) # Αντιστροφή για τον επόμενο παίκτη
            
            diff = score_before - score_after

            if board.is_checkmate():
                broadcast_speech("Δυστυχώς, είναι μάτ.")
            elif move_uci == best_move:
                broadcast_speech("Εξαιρετική κίνηση!")
            elif diff < 40:
                broadcast_speech("Πολύ καλή επιλογή.")
            elif diff < 150:
                broadcast_speech(f"Καλή κίνηση, αλλά η {best_move} ήταν καλύτερη.")
            else:
                broadcast_speech(f"Λάθος κίνηση. Έπρεπε να παίξεις {best_move}.")
        except: pass

    def get_score(self, turn):
        ev = self.sf.get_evaluation()
        val = ev['value']
        if ev['type'] == 'mate': val = 10000 if val > 0 else -10000
        return val if turn == chess.WHITE else -val

    def get_bot_move(self, board):
        # Εδώ το bot χρησιμοποιεί το δικό σου μοντέλο
        move_uci = None
        if self.ort_session:
            # ONNX Logic...
            pass
        if not move_uci:
            self.sf.set_fen_position(board.fen())
            move_uci = self.sf.get_best_move()
        return move_uci

# ==========================================
#          ΚΥΡΙΟΣ ΒΡΟΧΟΣ (LICHESS)
# ==========================================
def run_lichess_bot():
    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    brain = ChessBrain()
    me_id = client.account.get()['username'].lower()
    global last_analyzed_move_count

    print(f"🚀 Bot {me_id} is running and waiting for challenges...")

    for event in client.bots.stream_incoming_events():
        # ΑΥΤΟΜΑΤΗ ΑΠΟΔΟΧΗ
        if event['type'] == 'challenge':
            challenge_id = event['challenge']['id']
            print(f"⚔️ Αποδοχή πρόκλησης: {challenge_id}")
            client.bots.accept_challenge(challenge_id)

        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            print(f"🎮 Νέο παιχνίδι: {game_id}")
            stream = client.bots.stream_game_state(game_id)
            
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    bot_is_white = (g_evt['white'].get('id', '').lower() == me_id)
                    moves = g_evt['state']['moves'].split()
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                else: continue

                # Ενημέρωση Board
                board = chess.Board()
                for m in moves: board.push(chess.Move.from_uci(m))
                
                # Έλεγχος αν μόλις έπαιξε ο χρήστης
                current_move_count = len(moves)
                is_bot_turn = (board.turn == (chess.WHITE if bot_is_white else chess.BLACK))

                if is_bot_turn and not board.is_game_over() and current_move_count > 0:
                    if current_move_count > last_analyzed_move_count:
                        last_analyzed_move_count = current_move_count
                        user_move = moves[-1]
                        brain.analyze_move(board, user_move)
                        
                        # Απάντηση Bot
                        bot_move = brain.get_bot_move(board)
                        time.sleep(0.5)
                        client.bots.make_move(game_id, bot_move)

if __name__ == "__main__":
    # Ξεκινάμε το Flask (Site) σε ξεχωριστό thread
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)).start()
    # Ξεκινάμε το Bot
    run_lichess_bot()
