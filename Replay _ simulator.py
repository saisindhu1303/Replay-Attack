import json
import random
import time
import socket
import os
import hashlib

# === Configurations ===
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5002
LOCAL_DB = 'db.json'
SECRET_KEY = 'secret_key'  # Must match server

# === Load local database ===
def load_local_db():
    if not os.path.exists(LOCAL_DB):
        print(f"❌ Local DB not found at {LOCAL_DB}")
        return []
    with open(LOCAL_DB, 'r') as f:
        return json.load(f)

# === Generate token ===
def generate_token(user_id, timestamp, all_fvs):
    fv_str = "".join([str(fv["feature_vector"]) for fv in all_fvs])
    return hashlib.sha256(f"{user_id}:{timestamp}:{SECRET_KEY}:{fv_str}".encode()).hexdigest()

# === Replay Attack Function ===
def replay_attack(user_id=None, delay=0):
    db = load_local_db()
    if not db:
        return

    # Filter db entries by user_id field name (match your DB schema)
    if user_id:
        db = [entry for entry in db if entry.get('user_id', None) == user_id]

    if not db:
        print(f"❌ No records found for user '{user_id}'")
        return

    # Choose random replay entry
    replay_entry = random.choice(db)

    # Prepare all_feature_vectors for replay (based on db structure)
    all_feature_vectors = replay_entry.get("all_feature_vectors", None)
    if all_feature_vectors is None:
        # Fallback if DB uses single vector per user
        all_feature_vectors = [{
            "fe_index": 0,
            "feature_vector": replay_entry.get("feature_vector", [])
        }]

    timestamp = int(time.time())
    token = generate_token(replay_entry.get("user_id", "unknown"), timestamp, all_feature_vectors)

    payload = {
        "user_id": replay_entry.get("user_id", "unknown"),
        "timestamp": timestamp,
        "token": token,
        "all_feature_vectors": all_feature_vectors
    }

    data = json.dumps(payload).encode('utf-8')
    length_header = f"{len(data):<16}".encode()  # 16-byte length prefix

    # Send payload to server
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            s.sendall(length_header)
            s.sendall(data)
            response = s.recv(4096).decode()
            print(f"🎭 Replayed vector from '{payload['user_id']}' — Server response: {response}")
            time.sleep(delay)
    except Exception as e:
        print(f"❌ Error during replay: {e}")

# === Main ===
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Replay Attack Simulator")
    parser.add_argument('--user', type=str, help='User id to replay (optional)', default=None)
    parser.add_argument('--delay', type=float, help='Delay between attacks (seconds)', default=0.0)
    args = parser.parse_args()

    replay_attack(user_id=args.user, delay=args.delay)
