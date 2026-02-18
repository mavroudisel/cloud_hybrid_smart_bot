import berserk
import threading
import chess
import os
import time
import logging
import numpy as np
import onnxruntime as ort # <--- Η ΝΕΑ ΕΛΑΦΡΙΑ ΒΙΒΛΙΟΘΗΚΗ
from stockfish import Stockfish
from flask import Flask, render_template_string, jsonify

# ==========================================
#              ΡΥΘΜΙΣΕΙΣ
# ==========================================
TOKEN = os.environ.get("LICHESS_TOKEN") # Θα το πάρει από το Environment Variable
MODEL_ONNX = "my_chess_bot.onnx"
VOCAB_FILE = "vocab.npz"
STOCKFISH_PATH = "./stockfish"
BLUNDER_THRESHOLD = 70 

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

last_message = {"id": 0, "text": ""}
message_id = 0

def broadcast_speech(text):
    global last_message, message_id
    message_id += 1
    last_message = {"id": message_id, "text": text}
    print(f"🗣️ {text}")

@app.route('/')
def index(): return render_template_string("<h1>Coach Pro Active (ONNX Mode)</h1>")
@app.route('/poll')
def poll(): return jsonify(last_message)

def run_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ==========================================
#          ΥΒΡΙΔΙΚΟΣ ΕΓΚΕΦΑΛΟΣ (LIGHT)
# ==========================================
class HybridBrain:
    def __init__(self):
        # 1. Φόρτωση Stockfish (Μικρό Hash για μνήμη)
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Hash": 16})
        
        # 2. Φόρτωση ONNX (Ελαφρύ Μοντέλο)
        print("🧠 Loading ONNX Model...")
        try:
            # Φόρτωση Λεξικού
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            
            # Φόρτωση Μοντέλου
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
            print("✅ ONNX Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.ort_session = None

    def encode_board(self, board):
        # Μετατροπή board σε numpy array (1, 12, 8, 8)
        X = np.zeros((1, 12, 8, 8), dtype=np.float32)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            X[0, piece_map[piece.symbol()], 7 - rank, file] = 1
        return X

    def get_move(self, board):
        # Α. Πρόβλεψη ONNX
        my_move_uci = None
        if self.ort_session:
            try:
                input_feed = {self.ort_session.get_inputs()[0].name: self.encode_board(board)}
                output = self.ort_session.run(None, input_feed)[0]
                
                # Top κινήσεις (Softmax/Argmax logic απλοποιημένη)
                # Παίρνουμε τα indices με τα μεγαλύτερα scores
                top_indices = np.argsort(output[0])[::-1][:10]
                
                for idx in top_indices:
                    move_str = self.idx_to_move.get(idx)
                    if move_str and chess.Move.from_uci(move_str) in board.legal_moves:
                        my_move_uci = move_str
                        break
            except Exception as e:
                print(f"ONNX Error: {e}")

        if not my_move_uci:
            my_move_uci = self.sf.get_best_move()

        # Β. Έλεγχος Ασφαλείας (Stockfish)
        self.sf.set_fen_position(board.fen())
        best_uci = self.sf.get_best_move()

        if best_uci == my_move_uci:
            return best_uci

        self.sf.make_moves_from_current_position([best_uci])
        best_eval = self.get_eval()
        self.sf.set_fen_position(board.fen())

        self.sf.make_moves_from_current_position([my_move_uci])
        my_eval = self.get_eval()
        self.sf.set_fen_position(board.fen())

        loss = (best_eval - my_eval) if board.turn == chess.WHITE else (my_eval - best_eval)

        if loss > BLUNDER_THRESHOLD:
            broadcast_speech("Διόρθωσα λάθος σου.")
            return best_uci
        
        return my_move_uci

    def get_eval(self):
        e = self.sf.get_evaluation()
        return 10000 if e['type']=='mate' and e['value']>0 else (-10000 if e['type']=='mate' else e['value'])

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()

    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    brain = HybridBrain()
    me_id = client.account.get()['username'].lower()
    
    print(f"🚀 Bot Connected: {me_id}")

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.accept_challenge(event['challenge']['id'])
        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            print(f"New Game: {game_id}")
            stream = client.bots.stream_game_state(game_id)
            board = chess.Board()
            is_white = False
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    is_white = (g_evt['white']['id'].lower() == me_id)
                    moves = g_evt['state']['moves'].split()
                    board = chess.Board()
                    for m in moves: board.push(chess.Move.from_uci(m))
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                    board = chess.Board()
                    for m in moves: board.push(chess.Move.from_uci(m))

                if not board.is_game_over():
                    if (board.turn == chess.WHITE and is_white) or \
                       (board.turn == chess.BLACK and not is_white):
                        move = brain.get_move(board)
                        if move: client.bots.make_move(game_id, move)
