# utils.py

import base64
import hashlib
import json
import os
import time
from datetime import datetime

# === Token Generator ===
def generate_token(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

# === Timestamp ===
def get_timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# === Encode image to base64 ===
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === Decode base64 to image ===
def decode_image(encoded_str, save_path):
    img_bytes = base64.b64decode(encoded_str.encode("utf-8"))
    with open(save_path, "wb") as f:
        f.write(img_bytes)

# === Load Local DB ===
def load_local_db(path="db.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

# === Save Local DB ===
def save_local_db(data, path="db.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# === Delay Utility ===
def sleep(seconds):
    time.sleep(seconds)
