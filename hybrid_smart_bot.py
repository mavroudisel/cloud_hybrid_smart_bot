import berserk
import threading
import chess
import os
import time
import logging
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

# Χαμηλώνουμε το όριο για να διορθώνει πιο συχνά (ήταν 70)
BLUNDER_THRESHOLD = 30  

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Σύστημα Μηνυμάτων
last_message = {"id": 0, "text": ""}
message_id = 0

def broadcast_speech(text):
    global last_message, message_id
    message_id += 1
    last_message = {"id": message_id, "text": text}
    print(f"🗣️ AUDIO SENT: {text}")

# ==========================================
#          HTML ΓΙΑ ΤΟ ΚΙΝΗΤΟ (FIXED)
# ==========================================
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chess Coach</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; padding: 50px; }
            button { background-color: #4CAF50; color: white; padding: 20px 40px; font-size: 20px; border: none; border-radius: 10px; cursor: pointer; }
            #status { margin-top: 20px; color: #aaa; }
        </style>
    </head>
    <body>
        <h1>♟️ AI Coach Active</h1>
        <p>1. Πάτα το κουμπί παρακάτω.</p>
        <p>2. ΜΗΝ κλείσεις αυτή τη σελίδα (άσε την ανοιχτή).</p>
        <button onclick="startAudio()">🔊 ΕΝΕΡΓΟΠΟΙΗΣΗ ΗΧΟΥ</button>
        <div id="status">Αναμονή για εντολές...</div>

        <script>
            let lastId = 0;
            function startAudio() {
                // Dummy speak to unlock browser audio
                let utterance = new SpeechSynthesisUtterance("Audio System Online");
                window.speechSynthesis.speak(utterance);
                document.getElementById('status').innerText = "✅ Ο Ήχος Ενεργοποιήθηκε!";
                
                // Start polling
                setInterval(checkMessages, 1000);
            }

            function checkMessages() {
                fetch('/poll')
                .then(response => response.json())
                .then(data => {
                    if (data.id > lastId) {
                        lastId = data.id;
                        document.getElementById('status').innerText = "💬 " + data.text;
                        let msg = new SpeechSynthesisUtterance(data.text);
                        msg.lang = 'el-GR'; // Ελληνική φωνή
                        window.speechSynthesis.speak(msg);
                    }
                });
            }
        </script>
    </body>
    </html>
    """)

@app.route('/poll')
def poll(): return jsonify(last_message)

def run_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ==========================================
#          ΥΒΡΙΔΙΚΟΣ ΕΓΚΕΦΑΛΟΣ
# ==========================================
class HybridBrain:
    def __init__(self):
        # Stockfish Setup
        try:
            self.sf = Stockfish(path=STOCKFISH_PATH, depth=12, parameters={"Hash": 16, "Threads": 1})
            print("✅ Stockfish Loaded!")
        except Exception as e:
            print(f"❌ Stockfish Failed: {e}")

        # ONNX Setup
        print("🧠 Loading ONNX Model...")
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
            print("✅ ONNX Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ ONNX Failed: {e}")
            self.ort_session = None

    def encode_board(self, board):
        # Κωδικοποίηση 12x8x8
        X = np.zeros((1, 12, 8, 8), dtype=np.float32)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            # ΠΡΟΣΟΧΗ: Εδώ συνήθως γίνονται τα λάθη προσανατολισμού
            X[0, piece_map[piece.symbol()], 7 - rank, file] = 1
        return X

    def get_move(self, board):
        my_move_uci = None
        
        # --- 1. ΤΙ ΛΕΕΙ ΤΟ ΜΟΝΤΕΛΟ ΣΟΥ; ---
        if self.ort_session:
            try:
                input_feed = {self.ort_session.get_inputs()[0].name: self.encode_board(board)}
                output = self.ort_session.run(None, input_feed)[0]
                
                # Πάρε τις top 3 κινήσεις για debugging
                top_indices = np.argsort(output[0])[::-1][:3]
                print(f"📊 Model Top 3 predictions indices: {top_indices}")
                
                for idx in top_indices:
                    move_str = self.idx_to_move.get(idx)
                    if move_str:
                        move_obj = chess.Move.from_uci(move_str)
                        if move_obj in board.legal_moves:
                            print(f"🎯 Model picked legal move: {move_str}")
                            my_move_uci = move_str
                            break
                        else:
                            print(f"⚠️ Model picked ILLEGAL move: {move_str}")
            except Exception as e:
                print(f"⚠️ ONNX Error: {e}")

        # Fallback αν το μοντέλο απέτυχε πλήρως
        if not my_move_uci:
            print("⚠️ Model failed to give legal move. Using Stockfish as base.")
            my_move_uci = self.sf.get_best_move()

        # --- 2. ΤΙ ΛΕΕΙ Ο STOCKFISH (ΔΙΟΡΘΩΤΗΣ); ---
        try:
            self.sf.set_fen_position(board.fen())
            best_uci = self.sf.get_best_move()
            
            # Αξιολόγηση της κίνησης του Μοντέλου
            self.sf.make_moves_from_current_position([my_move_uci])
            my_eval = self.get_eval_score()
            
            # Επαναφορά και αξιολόγηση της τέλειας κίνησης
            self.sf.set_fen_position(board.fen())
            self.sf.make_moves_from_current_position([best_uci])
            best_eval = self.get_eval_score()
            self.sf.set_fen_position(board.fen()) # Reset

            # Υπολογισμός διαφοράς (πάντα θετική)
            # Centipawns: 100 = 1 πιόνι
            diff = abs(best_eval - my_eval)
            
            print(f"⚖️ Move Check: Mine({my_move_uci})={my_eval} vs Best({best_uci})={best_eval}. Diff={diff}")

            if diff > BLUNDER_THRESHOLD:
                print(f"🚨 BLUNDER DETECTED! Correcting {my_move_uci} -> {best_uci}")
                broadcast_speech("Διόρθωσα λάθος σου.")
                return best_uci
            
        except Exception as e:
            print(f"⚠️ Stockfish logic error: {e}")
            return best_uci # Fallback σε Stockfish αν χαλάσει ο κώδικας

        return my_move_uci

    def get_eval_score(self):
        # Επιστρέφει σκορ πάντα από την πλευρά του Λευκού για σύγκριση
        # ή απλά την απόλυτη τιμή της θέσης.
        ev = self.sf.get_evaluation()
        val = ev['value']
        if ev['type'] == 'mate':
            val = 10000 if val > 0 else -10000
        return val

# ==========================================
#          ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ
# ==========================================
if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()

    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    brain = HybridBrain()
    
    me_id = "unknown"
    try:
        me_id = client.account.get()['username'].lower()
        print(f"🚀 Bot Connected: {me_id}")
    except: pass

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.accept_challenge(event['challenge']['id'])
        
        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            print(f"🎮 New Game: {game_id}")
            
            stream = client.bots.stream_game_state(game_id)
            board = chess.Board()
            is_white = True
            
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    is_white = (g_evt['white'].get('id', '').lower() == me_id)
                    # Set initial state if needed
                    moves = g_evt['state']['moves'].split()
                    for m in moves: 
                        if m: board.push(chess.Move.from_uci(m))
                
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                    board = chess.Board()
                    for m in moves: 
                        if m: board.push(chess.Move.from_uci(m))

                if not board.is_game_over():
                    # Είναι η σειρά μου;
                    if board.turn == (chess.WHITE if is_white else chess.BLACK):
                        move = brain.get_move(board)
                        if move: 
                            client.bots.make_move(game_id, move)
