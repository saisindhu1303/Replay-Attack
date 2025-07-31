import cv2
import socket
import json
import numpy as np
import os
import time
import hashlib
import matplotlib.pyplot as plt
from feature_extractor import extract_feature_vector, load_feature_extractors

# === Configuration ===
SERVER_IP = '127.0.0.1'
SERVER_PORT = 5002
OUTPUT_DIR = "captured_images"
LOCAL_DB = "db.json"
COORD_FILE = "coord.txt"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SECRET_KEY = "secret_key"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Webcam Capture ===
def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return None
    print("📸 Press SPACE to capture image")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to capture", frame)
        if cv2.waitKey(1) == 32:
            cap.release()
            cv2.destroyAllWindows()
            return frame
    cap.release()
    cv2.destroyAllWindows()
    return None

# === Face Detection ===
def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# === Save Coordinates ===
def save_coords_to_file(faces, path=COORD_FILE):
    with open(path, "w") as f:
        for (x, y, w, h) in faces:
            f.write(f"{x} {y} {w} {h}\n")
    print(f"📝 Saved {len(faces)} face coordinates to {path}")

# === Cosine Similarity ===
def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load Local DB ===
def load_local_db():
    if os.path.exists(LOCAL_DB):
        with open(LOCAL_DB, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("❌ Invalid JSON in db.json")
    return []

# === Local DB Comparison supporting multi-FE DB ===
def compare_with_local_db(fv, db, threshold=0.85, fe_index=None):
    for entry in db:
        # Check if multi-FE vectors exist
        feature_vectors = entry.get("feature_vectors", None)
        if feature_vectors and fe_index is not None:
            db_fv = feature_vectors.get(str(fe_index))
        else:
            db_fv = entry.get("feature_vector")

        if not db_fv:
            continue
        if len(fv) != len(db_fv):
            continue

        sim = cosine_similarity(fv, db_fv)
        if sim >= threshold:
            return entry["user_id"], sim
    return None, 0.0

# === Manual LBP Feature Extraction ===
def extract_lbp_feature_vector(face_gray):
    if face_gray is None:
        return []
    radius = 1
    n_points = 8 * radius
    lbp_img = local_binary_pattern(face_gray, n_points, radius)
    hist, _ = np.histogram(lbp_img.ravel(), bins=np.arange(0, 257), density=True)
    return hist.tolist()

def local_binary_pattern(image, n_points, radius):
    from skimage.feature import local_binary_pattern
    return local_binary_pattern(image, n_points, radius, method="uniform").astype(np.uint8)

# === Graph Plotting ===
def plot_server_response(response_data):
    results = response_data.get("results", [])
    fe_indices = [str(r["fe_index"]) for r in results]
    similarities = [r["similarity"] for r in results]
    colors = ["green" if r["status"] == "granted" else "red" for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(fe_indices, similarities, color=colors)
    plt.axhline(0.85, color='blue', linestyle='--', label='Threshold = 0.85')
    plt.xlabel("Feature Extractor Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Feature Extractor Match Results")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# === Send to Server ===
def send_all_fvs_to_server(all_fvs, user_id="unknown"):
    timestamp = int(time.time())
    fv_str = "".join([str(fv["feature_vector"]) for fv in all_fvs])
    token = hashlib.sha256(f"{user_id}:{timestamp}:{SECRET_KEY}:{fv_str}".encode()).hexdigest()

    payload = {
        "user_id": user_id,
        "timestamp": timestamp,
        "token": token,
        "all_feature_vectors": all_fvs
    }

    data = json.dumps(payload).encode('utf-8')
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_IP, SERVER_PORT))
            s.sendall(f"{len(data):<16}".encode())
            s.sendall(data)
            response = s.recv(8192)
            response_obj = json.loads(response.decode())
            print("📬 Server response:", response_obj)

            with open("eval_results.json", "w") as f:
                json.dump(response_obj, f, indent=2)
            print("✅ Saved server response to eval_results.json")

            # Show the graph
            plot_server_response(response_obj)

    except Exception as e:
        print("❌ Server connection error:", e)

# === Main Flow ===
if __name__ == "__main__":
    image = capture_image()
    if image is None:
        exit()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        print("❌ No faces detected.")
        exit()

    save_coords_to_file(faces)
    db = load_local_db()
    fes = load_feature_extractors()
    if not fes:
        print("❌ No feature extractors loaded.")
        exit()

    for i, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]

        # Skip very small faces to avoid false matches
        if face.shape[0] < 50 or face.shape[1] < 50:
            print(f"⚠️ Face {i+1} too small (h={face.shape[0]}, w={face.shape[1]}), skipping")
            continue

        filename = os.path.join(OUTPUT_DIR, f"captured_face_{i+1}.png")
        cv2.imwrite(filename, face)
        print(f"🖼 Saved cropped face image: {filename}")

        all_fvs = []
        local_matches = []

        # LBP
        lbp_vector = extract_lbp_feature_vector(face)
        user_id_lbp, sim_lbp = compare_with_local_db(lbp_vector, db, threshold=0.85, fe_index=None)
        local_matches.append(("LBP", user_id_lbp, sim_lbp))
        all_fvs.append({
            "fe_index": "LBP",
            "feature_vector": lbp_vector
        })

        # Disposable FEs
        for fe_index in range(len(fes)):
            fv = extract_feature_vector(face, fe_index=fe_index)
            if not fv:
                print(f"❌ Failed to extract feature vector for FE {fe_index}")
                continue
            user_id, sim = compare_with_local_db(fv, db, threshold=0.85, fe_index=fe_index)
            local_matches.append((fe_index, user_id, sim))
            all_fvs.append({
                "fe_index": fe_index,
                "feature_vector": fv
            })

        # Print local match summary
        for fe_idx, uid, similarity in local_matches:
            if uid:
                print(f"✅ FE {fe_idx}: Local match found - User ID: {uid}, Similarity: {similarity:.2f}")
            else:
                print(f"⚠️ FE {fe_idx}: No local match.")

        # Send to server
        send_all_fvs_to_server(all_fvs, user_id="unknown")
