import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load Evaluation Data ===
EVAL_RESULTS_FILE = "eval_results.json"

# === Helper Functions ===
def load_results():
    with open(EVAL_RESULTS_FILE, "r") as f:
        return json.load(f)

def plot_similarity_distributions(results):
    similarities = [r['similarity'] for r in results if r['similarity'] > 0]
    labels = [f"FE {r['fe_index']}\n{r['matched_user']}" for r in results if r['similarity'] > 0]

    plt.figure(figsize=(16, 6))
    sns.barplot(x=labels, y=similarities)
    plt.xticks(rotation=90)
    plt.ylabel("Cosine Similarity")
    plt.title("Similarity Score per Feature Extractor")
    plt.tight_layout()
    plt.show()

def plot_granted_vs_denied(results):
    granted = [r['fe_index'] for r in results if r['status'] == 'granted']
    denied = [r['fe_index'] for r in results if r['status'] == 'denied']

    counts = [granted.count(i) for i in range(30)]
    denied_counts = [denied.count(i) for i in range(30)]

    x = np.arange(30)
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, counts, 0.4, label='Granted')
    plt.bar(x + 0.2, denied_counts, 0.4, label='Denied')
    plt.xlabel("Feature Extractor Index")
    plt.ylabel("Count")
    plt.title("Grant vs Deny Decisions per FE")
    plt.legend()
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

def show_summary(results):
    granted_count = sum(1 for r in results if r['status'] == 'granted')
    total = len(results)
    print(f"✅ Access Granted by {granted_count}/{total} FEs")

    similarities = [r['similarity'] for r in results if r['similarity'] > 0]
    print(f"📊 Avg Similarity: {np.mean(similarities):.4f}, Max: {np.max(similarities):.4f}, Min: {np.min(similarities):.4f}")

# === Main Execution ===
if __name__ == "__main__":
    if not os.path.exists(EVAL_RESULTS_FILE):
        print("❌ eval_results.json not found. Run client-server interaction first.")
        exit()

    results = load_results()["results"]
    show_summary(results)
    plot_similarity_distributions(results)
    plot_granted_vs_denied(results)
