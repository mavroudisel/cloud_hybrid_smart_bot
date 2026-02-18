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

# Όρια Αξιολόγησης (σε εκατοστά του πιονιού - centipawns)
EVAL_TOLERANCE = 50    # Μέχρι 0.5 πιόνι διαφορά = "Καλό"
BLUNDER_LIMIT = 200    # Πάνω από 2 πιόνια διαφορά = "Blunder"

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

last_message = {"id": 0, "text": ""}
message_id = 0

def broadcast_speech(text):
    global last_message, message_id
    message_id += 1
    last_message = {"id": message_id, "text": text}
    print(f"🗣️ AUDIO SENT: {text}")

# ==========================================
#          HTML INTERFACE (MOBILE)
# ==========================================
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Chess Coach</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #1a1a1a; color: #ddd; font-family: 'Arial', sans-serif; text-align: center; padding: 20px; }
            h1 { color: #4CAF50; }
            button { background-color: #ff9800; color: white; padding: 25px 50px; font-size: 22px; border: none; border-radius: 12px; cursor: pointer; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            button:active { transform: translateY(2px); }
            #last-msg { font-size: 26px; font-weight: bold; color: #fff; margin-top: 20px; padding: 20px; border: 2px solid #555; border-radius: 10px; background: #333; min-height: 50px;}
            .info { font-size: 14px; color: #888; margin-top: 50px; }
        </style>
    </head>
    <body>
        <h1>♟️ Προπονητής Live</h1>
        <p>Θα σχολιάζω κάθε σου κίνηση.</p>
        <button onclick="startAudio()">🔊 ΕΝΕΡΓΟΠΟΙΗΣΗ ΗΧΟΥ</button>
        <div id="last-msg">Αναμονή...</div>
        <div class="info">Κράτα αυτή τη σελίδα ανοιχτή στο κινητό.</div>

        <script>
            let lastId = 0;
            function startAudio() {
                let u = new SpeechSynthesisUtterance("Σύστημα ανάλυσης έτοιμο.");
                u.lang = 'el-GR';
                window.speechSynthesis.speak(u);
                document.getElementById('last-msg').innerText = "✅ Συνδέθηκε";
                document.getElementById('last-msg').style.borderColor = "#4CAF50";
                setInterval(checkMessages, 1000);
            }

            function checkMessages() {
                fetch('/poll')
                .then(r => r.json())
                .then(data => {
                    if (data.id > lastId) {
                        lastId = data.id;
                        document.getElementById('last-msg').innerText = data.text;
                        
                        // Speak
                        window.speechSynthesis.cancel(); // Stop previous
                        let msg = new SpeechSynthesisUtterance(data.text);
                        msg.lang = 'el-GR';
                        msg.rate = 1.1; 
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
#          Η ΝΟΗΜΟΣΥΝΗ (ΔΥΟ ΜΕΡΗ)
# ==========================================
class ChessBrain:
    def __init__(self):
        # Stockfish (Δυνατός για ανάλυση)
        self.sf = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Hash": 64})
        
        # ONNX Model (Το στυλ σου για να παίζει το bot)
        print("🧠 Loading ONNX Model...")
        try:
            data = np.load(VOCAB_FILE, allow_pickle=True)
            self.vocab = data['vocab'].item()
            self.idx_to_move = {v: k for k, v in self.vocab.items()}
            self.ort_session = ort.InferenceSession(MODEL_ONNX)
            print("✅ ONNX Loaded!")
        except:
            self.ort_session = None

    def analyze_user_move(self, board, move_uci):
        """
        Αναλύει την κίνηση που ΜΟΛΙΣ έκανε ο χρήστης.
        Συγκρίνει την κίνηση χρήστη με την τέλεια κίνηση του Stockfish.
        """
        try:
            # 1. Πηγαίνουμε το Board ΜΙΑ κίνηση πίσω (πριν παίξει ο χρήστης)
            # για να δούμε τι επιλογές είχε.
            board.pop() 
            self.sf.set_fen_position(board.fen())
            
            # 2. Βρες την τέλεια κίνηση
            best_move = self.sf.get_best_move()
            
            # 3. Αξιολόγησε την κίνηση του χρήστη
            self.sf.make_moves_from_current_position([move_uci])
            user_eval = self._get_eval_val()
            
            # 4. Αξιολόγησε την τέλεια κίνηση
            self.sf.set_fen_position(board.fen()) # Reset
            self.sf.make_moves_from_current_position([best_move])
            best_eval = self._get_eval_val()
            
            # 5. Επαναφορά Board στην τρέχουσα κατάσταση (για να συνεχίσει το παιχνίδι)
            board.push(chess.Move.from_uci(move_uci))

            # 6. Υπολογισμός διαφοράς
            # Η διαφορά είναι πάντα θετική (πόσο χειρότερη είναι η κίνηση του user από την best)
            diff = abs(best_eval - user_eval)

            # 7. ΣΧΟΛΙΑΣΜΟΣ
            if move_uci == best_move or diff < 20:
                broadcast_speech("Άριστη κίνηση!")
            elif diff < EVAL_TOLERANCE:
                broadcast_speech("Πολύ καλή κίνηση.")
            elif diff < BLUNDER_LIMIT:
                broadcast_speech("Μέτρια κίνηση. Υπήρχε και καλύτερη.")
            else:
                broadcast_speech(f"Πρόσεξε! Αυτό ήταν λάθος. Καλύτερη ήταν η {best_move}.")
                
        except Exception as e:
            print(f"Analysis Error: {e}")
            # Αν χαλάσει η ανάλυση, δεν λέμε τίποτα για να μην μπερδέψουμε
            pass

    def get_bot_move(self, board):
        """
        Βρίσκει τι θα παίξει το BOT (μίμηση εσένα + blunder check)
        """
        my_move_uci = None
        
        # Α. Μίμηση (ONNX)
        if self.ort_session:
            try:
                X = self._encode_board(board)
                input_name = self.ort_session.get_inputs()[0].name
                output = self.ort_session.run(None, {input_name: X})[0]
                top_indices = np.argsort(output[0])[::-1][:5]
                for idx in top_indices:
                    m = self.idx_to_move.get(idx)
                    if m and chess.Move.from_uci(m) in board.legal_moves:
                        my_move_uci = m
                        break
            except: pass
        
        if not my_move_uci: 
            my_move_uci = self.sf.get_best_move()

        # Β. Έλεγχος Blunder Bot (να μην παίζει χάλια το bot)
        self.sf.set_fen_position(board.fen())
        best_uci = self.sf.get_best_move()
        
        if my_move_uci == best_uci: return my_move_uci
        
        # Check diff
        self.sf.make_moves_from_current_position([my_move_uci])
        eval_mine = self._get_eval_val()
        self.sf.set_fen_position(board.fen())
        self.sf.make_moves_from_current_position([best_uci])
        eval_best = self._get_eval_val()
        
        # Το bot επιτρέπεται να παίζει λίγο χειρότερα (για να έχει το στυλ σου)
        # αλλά όχι τραγικά (250 cp limit).
        if abs(eval_best - eval_mine) > 250:
            return best_uci
        return my_move_uci

    def _get_eval_val(self):
        ev = self.sf.get_evaluation()
        val = ev['value']
        if ev['type'] == 'mate': val = 10000 if val > 0 else -10000
        return val

    def _encode_board(self, board):
        X = np.zeros((1, 12, 8, 8), dtype=np.float32)
        piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
        for sq, pc in board.piece_map().items():
            r, f = divmod(sq, 8)
            X[0, piece_map[pc.symbol()], 7 - r, f] = 1
        return X

# ==========================================
#          MAIN LOOP
# ==========================================
if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()

    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)
    brain = ChessBrain()

    try:
        me_id = client.account.get()['username'].lower()
        print(f"🚀 Coach Ready: {me_id}")
    except: pass

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.accept_challenge(event['challenge']['id'])
        
        elif event['type'] == 'gameStart':
            game_id = event['game']['gameId']
            stream = client.bots.stream_game_state(game_id)
            board = chess.Board()
            is_white_bot = True # Υποθέτουμε αρχικά ότι το bot είναι White
            
            for g_evt in stream:
                if g_evt['type'] == 'gameFull':
                    # Ποιος είναι το Bot;
                    is_white_bot = (g_evt['white'].get('id', '').lower() == me_id)
                    
                    # Φόρτωσε κινήσεις που έγιναν ήδη
                    moves = g_evt['state']['moves'].split()
                    for m in moves: 
                        if m: board.push(chess.Move.from_uci(m))
                
                elif g_evt['type'] == 'gameState':
                    moves = g_evt['moves'].split()
                    # Ξαναφτιάχνουμε το board για σιγουριά
                    board = chess.Board()
                    for m in moves: 
                        if m: board.push(chess.Move.from_uci(m))
                    
                    # --- ΕΔΩ ΕΙΝΑΙ Η ΑΛΛΑΓΗ ---
                    # Μόλις ήρθε νέο state. Ποιος έπαιξε τελευταίος;
                    # Αν τώρα είναι σειρά του Bot, σημαίνει ότι ΜΟΛΙΣ ΕΠΑΙΞΕ Ο ΧΡΗΣΤΗΣ.
                    if board.turn == (chess.WHITE if is_white_bot else chess.BLACK):
                        if len(moves) > 0:
                            last_move = moves[-1] # Η κίνηση που μόλις έκανες εσύ
                            print(f"👀 User played: {last_move}. Analyzing...")
                            brain.analyze_user_move(board, last_move)

                # Αν το παιχνίδι δεν τελείωσε και είναι σειρά του Bot
                if not board.is_game_over():
                    if board.turn == (chess.WHITE if is_white_bot else chess.BLACK):
                        # Το Bot σκέφτεται την απάντησή του
                        bot_move = brain.get_bot_move(board)
                        if bot_move: 
                            client.bots.make_move(game_id, bot_move)
