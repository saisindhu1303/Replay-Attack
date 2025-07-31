import socket
import json
import os
import threading
import numpy as np
import hashlib

DB_FILE = "db.json"
RECEIVED_DIR = "received_vectors"
SERVER_IP = "0.0.0.0"
SERVER_PORT = 5002
SECRET_KEY = "secret_key"  # Must match client secret

os.makedirs(RECEIVED_DIR, exist_ok=True)

# === Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load DB ===
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("❌ db.json format error")
    return []

# === Match FV to DB for specific FE index ===
def match_user(fv, db, fe_index, threshold=0.85):
    best_match = None
    best_sim = 0.0
    for entry in db:
        feature_vectors = entry.get("feature_vectors", {})
        db_fv = feature_vectors.get(str(fe_index))
        if db_fv is None:
            continue
        if len(fv) != len(db_fv):
            continue
        sim = cosine_similarity(fv, db_fv)
        if sim > best_sim and sim >= threshold:
            best_sim = sim
            best_match = entry["user_id"]
    return best_match, best_sim

# === Validate token for combined payload ===
def validate_token(user_id, timestamp, token, all_fvs, secret_key=SECRET_KEY):
    fv_string = "".join([str(fv["feature_vector"]) for fv in all_fvs])
    expected = hashlib.sha256(f"{user_id}:{timestamp}:{secret_key}:{fv_string}".encode()).hexdigest()
    return expected == token

# === Handle Incoming Client ===
def handle_client(conn, addr):
    print(f"📥 Connection from {addr}")

    try:
        length_data = conn.recv(16)
        if not length_data:
            print("❌ No data received.")
            return

        total_length = int(length_data.decode().strip())
        chunks = []
        received = 0

        while received < total_length:
            chunk = conn.recv(min(4096, total_length - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)

        data = b''.join(chunks)
        obj = json.loads(data.decode())

        user_id = obj.get("user_id", "unknown")
        timestamp = obj.get("timestamp", 0)
        token = obj.get("token", "")
        all_fvs = obj.get("all_feature_vectors", [])

        if not all_fvs:
            response = {"status": "denied", "reason": "no_feature_vectors"}
            conn.sendall(json.dumps(response).encode())
            return

        # Validate combined token
        if not validate_token(user_id, timestamp, token, all_fvs):
            print("❌ Invalid token from client.")
            response = {"status": "denied", "reason": "invalid_token"}
            conn.sendall(json.dumps(response).encode())
            return

        db = load_db()
        results = []

        for fe_entry in all_fvs:
            fe_index = fe_entry.get("fe_index", -1)
            fv = fe_entry.get("feature_vector", [])

            matched_id, sim = match_user(fv, db, fe_index)
            results.append({
                "fe_index": fe_index,
                "status": "granted" if matched_id else "denied",
                "matched_user": matched_id if matched_id else "unknown",
                "similarity": round(sim, 4)
            })

            # Save each feature vector for auditing
            save_path = os.path.join(RECEIVED_DIR, f"user_{user_id}_fe{fe_index}_{timestamp}.json")
            with open(save_path, "w") as f:
                json.dump(fe_entry, f, indent=2)

        final_decision = "granted" if any(r["status"] == "granted" for r in results) else "denied"

        response = {
            "results": results,
            "final_decision": final_decision
        }

        conn.sendall(json.dumps(response).encode())
        print(f"✅ Response sent: {response}")

    except Exception as e:
        print("❌ Error handling client:", e)
    finally:
        conn.close()

# === Start Server ===
def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SERVER_IP, SERVER_PORT))
        s.listen()
        print(f"🚀 Server listening on {SERVER_IP}:{SERVER_PORT} ...")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr)).start()

# === Entry Point ===
if __name__ == "__main__":
    start_server()
