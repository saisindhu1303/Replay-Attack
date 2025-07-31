import json
import os

DB_FILE = "db.json"

def clean_db(file_path=DB_FILE):
    if not os.path.exists(file_path):
        print(f"No database file found at {file_path}")
        return

    with open(file_path, "r") as f:
        db = json.load(f)

    original_len = len(db)
    cleaned_db = [entry for entry in db if len(entry["feature_vector"]) == 60]

    with open(file_path, "w") as f:
        json.dump(cleaned_db, f, indent=2)

    removed = original_len - len(cleaned_db)
    print(f"✅ Cleaned database. Removed {removed} invalid entr{'y' if removed == 1 else 'ies'}.")

if __name__ == "__main__":
    clean_db()
