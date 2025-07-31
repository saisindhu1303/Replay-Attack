import cv2
import os
import json
import time
from feature_extractor import extract_feature_vector

# === Configuration ===
CAPTURED_DIR = "captured_images"
LOCAL_DB = "db.json"
FE_INDEX = 0  # Patch-based feature extractor index (0-29)

os.makedirs(CAPTURED_DIR, exist_ok=True)

def save_local_db(db):
    with open(LOCAL_DB, "w") as f:
        json.dump(db, f, indent=2)
    print(f"💾 Saved database with {len(db)} users.")

def load_local_db():
    if os.path.exists(LOCAL_DB):
        with open(LOCAL_DB, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("❌ Invalid JSON in db.json, starting fresh.")
    return []

def capture_face():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return None

    print("📸 Press SPACE to capture your face image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        cv2.imshow("Enroll - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE pressed
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == 27:  # ESC pressed
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    print("=== Enroll New User ===")
    user_id = input("Enter user ID (e.g., user123): ").strip()
    if not user_id:
        print("❌ User ID cannot be empty.")
        return

    image = capture_face()
    if image is None:
        print("❌ No image captured.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # For enrollment, you might want to detect face and crop it here
    # For simplicity, let's assume entire gray image is face
    face = gray

    fv = extract_feature_vector(face, fe_index=FE_INDEX)
    if not fv:
        print("❌ Failed to extract feature vector.")
        return

    db = load_local_db()
    # Add new user or update existing
    updated = False
    for entry in db:
        if entry.get("user_id") == user_id:
            entry["feature_vector"] = fv
            updated = True
            print(f"🔄 Updated feature vector for user {user_id}.")
            break

    if not updated:
        db.append({
            "user_id": user_id,
            "feature_vector": fv
        })
        print(f"➕ Added new user {user_id} to database.")

    save_local_db(db)
    # Save captured face image as record
    timestamp = int(time.time())
    filename = os.path.join(CAPTURED_DIR, f"{user_id}_{timestamp}.png")
    cv2.imwrite(filename, face)
    print(f"🖼 Saved captured face image to {filename}")

if __name__ == "__main__":
    main()
