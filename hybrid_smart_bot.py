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
last_analyzed_move = "" # Για να μην μιλάει δύο φορές για την ίδια κίνηση

def broadcast_speech(text):
    global last_message, message_id
    message_id += 1
    last_message = {"id": message_id, "text": text}
    print(f"🗣️ Σχόλιο: {text}")

@app.route('/')
def index(): return render_template_string("<h1>Coach Active</h1>")
@app.route('/poll')
def poll(): return jsonify(last_message)

# ==========================================
#          ΕΓΚΕΦΑΛΟΣ ΑΝΑΛΥΣΗΣ
# ==========================================
class ChessCoach:
    def __init__(self):
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=15)
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
        except: self.ort_session = None

    def analyze_user_move(self, board, move_uci):
        global last_analyzed_move
        if move_uci == last_analyzed_move: return # Μην αναλύεις την ίδια κίνηση
        last_analyzed_move = move_uci

        try:
            # Παίρνουμε τη θέση ΠΡΙΝ την κίνησή σου
            board.pop()
            fen_before = board.fen()
            self.sf.set_fen_position(fen_before)
            best_move = self.sf.get_best_move()
            
            # Αξιολόγηση πριν και μετά (από την πλευρά του παίκτη)
            eval_before = self.get_score(board.turn)
            
            board.push(chess.Move.from_uci(move_uci))
            self.sf.set_fen_position(board.fen())
            # Αξιολόγηση μετά (αντιστρέφουμε γιατί άλλαξε η σειρά, αλλά θέλουμε το σκορ του παίκτη)
            eval_after = -self.get_score(board.turn) 

            diff = eval_before - eval_after

            if board.is_checkmate():
                broadcast_speech("Δυστυχώς, αυτό είναι ματ.")
            elif move_uci == best_move:
                broadcast_speech("Άριστη κίνηση, ακριβώς όπως ο Stockfish!")
            elif diff < 30:
                broadcast_speech("Πολύ καλή κίνηση.")
            elif diff < 100:
                broadcast_speech(f"Καλή κίνηση, αλλά η {best_move} ήταν ελαφρώς καλύτερη.")
            elif diff < 300:
                broadcast_speech(f"Μέτρια κίνηση. Σοβαρή εναλλακτική ήταν η {best_move}.")
            else:
                broadcast_speech(f"Αυτό είναι μεγάλο λάθος. Η κίνηση {best_move} ήταν πολύ καλύτερη.")

        except Exception as e: print(f"Error: {e}")

    def get_score(self, turn):
        ev = self.sf.get_evaluation()
        val = ev['value']
        if ev['type'] == 'mate': val = 10000 if val > 0 else -10000
        # Επιστρέφει σκορ θετικό αν είναι καλό για αυτόν που παίζει
        return val if turn == chess.WHITE else -val

    def get_bot_move(self, board):
        # Εδώ το bot παίζει με το μοντέλο σου
        move_uci = None
        if self.ort_session:
            # (Κώδικας ONNX...)
            pass
        if not move_uci:
            self.sf.set_fen_position(board.fen())
            move_uci = self.sf.get_best_move()
        return move_uci

# ==========================================
#          ΚΥΡΙΟΣ ΒΡΟΧΟΣ
# ==========================================
def main_loop():
    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    coach = ChessCoach()
    me_id = client.account.get()['username'].lower()

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            stream = client.bots.stream_game_state(game_id)
            board = chess.Board()
            
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    bot_is_white = (g_evt['white'].get('id', '').lower() == me_id)
                    moves = g_evt['state']['moves'].split()
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                else: continue

                # Ενημέρωση σκακιέρας
                board = chess.Board()
                for m in moves: board.push(chess.Move.from_uci(m))

                # ΛΟΓΙΚΗ: Αν είναι η σειρά του BOT, σημαίνει ότι ο ΧΡΗΣΤΗΣ μόλις έπαιξε
                is_bot_turn = (board.turn == (chess.WHITE if bot_is_white else chess.BLACK))
                
                if is_bot_turn and not board.is_game_over() and len(moves) > 0:
                    user_last_move = moves[-1]
                    coach.analyze_user_move(board, user_last_move)
                    
                    # Το bot απαντάει
                    bot_move = coach.get_bot_move(board)
                    time.sleep(1) # Μικρή καθυστέρηση για να ακουστεί το σχόλιο
                    client.bots.make_move(game_id, bot_move)

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=10000)).start()
    main_loop()
