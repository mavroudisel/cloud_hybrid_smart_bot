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
def index(): return render_template_string("<h1>Coach Pro Active (v2 Robust)</h1>")
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
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Hash": 16})
        print("🧠 Loading ONNX Model...")
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
            print("✅ ONNX Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.ort_session = None

    def encode_board(self, board):
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
        my_move_uci = None
        # 1. Πρόβλεψη Neural Network
        if self.ort_session:
            try:
                input_feed = {self.ort_session.get_inputs()[0].name: self.encode_board(board)}
                output = self.ort_session.run(None, input_feed)[0]
                top_indices = np.argsort(output[0])[::-1][:10]
                for idx in top_indices:
                    move_str = self.idx_to_move.get(idx)
                    if move_str and chess.Move.from_uci(move_str) in board.legal_moves:
                        my_move_uci = move_str
                        break
            except Exception as e:
                print(f"⚠️ ONNX Error: {e}")

        if not my_move_uci:
            my_move_uci = self.sf.get_best_move()

        # 2. Έλεγχος Stockfish (Blunder Check)
        try:
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

            # Υπολογισμός διαφοράς (White or Black perspective)
            eval_diff = best_eval - my_eval if board.turn == chess.WHITE else my_eval - best_eval

            if eval_diff > BLUNDER_THRESHOLD:
                broadcast_speech("Διόρθωσα λάθος σου.")
                return best_uci
        except Exception as e:
            print(f"⚠️ Stockfish Error: {e}")
            return my_move_uci or best_uci
        
        return my_move_uci

    def get_eval(self):
        e = self.sf.get_evaluation()
        return 10000 if e['type']=='mate' and e['value']>0 else (-10000 if e['type']=='mate' else e['value'])

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
    
    # Λήψη Username με ασφάλεια
    try:
        me = client.account.get()
        me_id = me['username'].lower()
        print(f"🚀 Bot Connected: {me_id}")
    except Exception as e:
        print("❌ Token Error. Check Environment Variables.")
        me_id = "unknown"

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            # Αποδοχή μόνο Standard & Casual για αρχή
            if event['challenge']['variant']['key'] == 'standard':
                print(f"⚔️ Accepting Challenge: {event['challenge']['id']}")
                client.bots.accept_challenge(event['challenge']['id'])
            else:
                print("🚫 Declined non-standard challenge")
                client.bots.decline_challenge(event['challenge']['id'])
        
        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            print(f"🎮 New Game Started: {game_id}")
            
            stream = client.bots.stream_game_state(game_id)
            board = chess.Board()
            is_white = True # Default assumption
            
            for g_evt in stream:
                try:
                    if g_evt['type'] == 'gameFull':
                        # 1. Ρύθμιση Χρώματος
                        white_player = g_evt['white'].get('id', '').lower()
                        is_white = (white_player == me_id)
                        print(f"ℹ️ Playing as: {'WHITE' if is_white else 'BLACK'}")

                        # 2. Ρύθμιση Αρχικής Θέσης (για FEN/Variants)
                        initial_fen = g_evt.get('initialFen')
                        if initial_fen and initial_fen != 'startpos':
                            board = chess.Board(initial_fen)
                        else:
                            board = chess.Board()

                        # 3. Ενημέρωση Κινήσεων
                        moves = g_evt['state']['moves'].split()
                        for m in moves: 
                            if m: board.push(chess.Move.from_uci(m))

                    elif g_evt['type'] == 'gameState':
                        moves = g_evt['moves'].split()
                        # Rebuild board to be safe
                        # (Απλοϊκός τρόπος για σιγουριά)
                        board = chess.Board() 
                        for m in moves: 
                            if m: board.push(chess.Move.from_uci(m))

                    # 4. Λογική Κίνησης
                    if not board.is_game_over():
                        my_turn = (board.turn == chess.WHITE and is_white) or \
                                  (board.turn == chess.BLACK and not is_white)
                        
                        if my_turn:
                            print("🤔 Thinking...")
                            move = brain.get_move(board)
                            if move:
                                print(f"👉 Playing: {move}")
                                client.bots.make_move(game_id, move)
                
                except Exception as e:
                    # Η ΑΣΠΙΔΑ: Αν γίνει λάθος, το γράφει και συνεχίζει!
                    print(f"⚠️ Game Error (Ignored): {e}")
                    continue
