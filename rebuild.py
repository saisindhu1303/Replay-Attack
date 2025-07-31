import os
import json
import cv2
from feature_extractor import extract_feature_vector

# === Configuration ===
CAPTURED_IMAGES_DIR = "captured_images"  # Folder with face images
OUTPUT_DB = "db.json"             # Output JSON database file
MAX_FE_INDEX = 29                         # Number of FEs (0 to 29)

def get_user_id_from_filename(filename):
    """
    Extract user ID from filename by removing timestamp or image index.
    Skips generic images like 'captured_face_1.png' or 'patch_01.png'.
    """
    fname = filename.lower()
    if any(skip in fname for skip in ["captured", "patch", "face"]):
        return None  # Skip generic or testing images

    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) >= 2:
        return "_".join(parts[:-1])  # everything except last part (timestamp or index)
    else:
        return base

def rebuild_local_db_all_fes():
    db = {}
    files = [f for f in os.listdir(CAPTURED_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} face images in {CAPTURED_IMAGES_DIR}")

    for f in files:
        user_id = get_user_id_from_filename(f)
        if not user_id:
            print(f"⏩ Skipped {f} (no user ID)")
            continue

        path = os.path.join(CAPTURED_IMAGES_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Failed to load {path}")
            continue

        if user_id not in db:
            db[user_id] = {"user_id": user_id, "feature_vectors": {}}

        for fe_index in range(MAX_FE_INDEX + 1):
            fv = extract_feature_vector(img, fe_index=fe_index)
            if fv:
                db[user_id]["feature_vectors"][str(fe_index)] = fv
                print(f"✅ FE {fe_index} processed for {user_id}, FV length={len(fv)}")
            else:
                print(f"⚠️ FE {fe_index} failed to extract FV for {f}")

    db_list = list(db.values())
    with open(OUTPUT_DB, "w") as outfile:
        json.dump(db_list, outfile, indent=2)
    print(f"🎉 Saved updated DB with {len(db_list)} users and multiple FEs to {OUTPUT_DB}")

if __name__ == "__main__":
    rebuild_local_db_all_fes()
