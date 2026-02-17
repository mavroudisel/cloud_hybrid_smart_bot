import berserk
import threading
import chess
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from stockfish import Stockfish
from flask import Flask, render_template_string, jsonify

# ==========================================
#              ΡΥΘΜΙΣΕΙΣ
# ==========================================
TOKEN = os.environ.get("LICHESS_TOKEN", "lip_wUwKcLiYjvOkSBB9wN9R")
MODEL_PATH = "my_chess_bot.pth"
STOCKFISH_PATH = "./stockfish"
BLUNDER_THRESHOLD = 70  # Αυστηρότητα (σε centipawns)

# ==========================================
#          WEB SERVER (FLASK)
# ==========================================
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
def index(): return render_template_string("<h1>Coach Pro Active</h1>")
@app.route('/poll')
def poll(): return jsonify(last_message)

def run_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ==========================================
#      ΑΡΧΙΤΕΚΤΟΝΙΚΗ ΜΟΝΤΕΛΟΥ (ΑΠΟ TRAIN.PY)
# ==========================================
class ChessNet(nn.Module):
    def __init__(self, num_moves):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_moves)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
#          ΥΒΡΙΔΙΚΟΣ ΕΓΚΕΦΑΛΟΣ
# ==========================================
class HybridBrain:
    def __init__(self):
        # 1. Φόρτωση Stockfish
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Hash": 64})
        
        # 2. Φόρτωση PyTorch Μοντέλου
        print("🧠 Loading PyTorch Model...")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            self.vocab = checkpoint['vocab']
            # Αντιστροφή λεξικού (από νούμερο σε κίνηση)
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            
            num_moves = len(self.vocab)
            self.model = ChessNet(num_moves)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval() # Mode για παιχνίδι (όχι εκπαίδευση)
            print("✅ Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def encode_board(self, board):
        # Μετατροπή board σε 12x8x8 Tensor (όπως στην εκπαίδευση)
        X = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            X[piece_map[piece.symbol()], 7 - rank, file] = 1
        return torch.tensor(X).unsqueeze(0) # Add batch dimension

    def get_move(self, board):
        # Α. Προσπάθεια πρόβλεψης από το ΔΙΚΟ ΣΟΥ μοντέλο
        my_move_uci = None
        if self.model:
            try:
                input_tensor = self.encode_board(board)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    # Παίρνουμε τις top 5 πιθανές κινήσεις (για να βρούμε μια νόμιμη)
                    _, top_indices = torch.topk(output, 10)
                    
                # Βρες την πρώτη ΝΟΜΙΜΗ κίνηση που προτείνει το μοντέλο
                for idx in top_indices[0]:
                    move_str = self.idx_to_move.get(idx.item())
                    if move_str and chess.Move.from_uci(move_str) in board.legal_moves:
                        my_move_uci = move_str
                        break
            except Exception as e:
                print(f"Model Error: {e}")

        # Αν το μοντέλο απέτυχε ή δεν βρήκε νόμιμη, βάλε Stockfish
        if not my_move_uci:
            my_move_uci = self.sf.get_best_move()

        # Β. Έλεγχος Ασφαλείας (Safety Net)
        self.sf.set_fen_position(board.fen())
        best_uci = self.sf.get_best_move()

        if best_uci == my_move_uci:
            return best_uci

        # Σύγκριση
        self.sf.make_moves_from_current_position([best_uci])
        best_eval = self.get_eval()
        self.sf.set_fen_position(board.fen())

        self.sf.make_moves_from_current_position([my_move_uci])
        my_eval = self.get_eval()
        self.sf.set_fen_position(board.fen())

        loss = (best_eval - my_eval) if board.turn == chess.WHITE else (my_eval - best_eval)

        if loss > BLUNDER_THRESHOLD:
            broadcast_speech("Διόρθωσα λάθος σου.")
            print(f"⚠️ SAFETY NET: Loss {loss}")
            return best_uci
        
        return my_move_uci

    def get_eval(self):
        e = self.sf.get_evaluation()
        return 10000 if e['type']=='mate' and e['value']>0 else (-10000 if e['type']=='mate' else e['value'])

# ==========================================
#          MAIN LOOP
# ==========================================
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
                        if move:
                            client.bots.make_move(game_id, move)