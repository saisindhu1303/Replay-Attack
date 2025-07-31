import numpy as np
import cv2
from feature_extractor import extract_feature_vector, load_feature_extractors
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern

def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    if a.shape != b.shape:
        # Cannot compute similarity for vectors of different size
        return np.nan
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# LBP feature extraction (same as before)
def extract_lbp_feature_vector(gray_image, radius=1, n_points=8):
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

# Load image and extract features (same as before)
TEST_IMAGE_PATH = "captured_images/captured_face_1.png"
image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("❌ Could not load test image:", TEST_IMAGE_PATH)
    exit()

lbp_vector = extract_lbp_feature_vector(image)
fes = load_feature_extractors()
if not fes:
    print("❌ No disposable feature extractors found.")
    exit()
disposable_vectors = [extract_feature_vector(image, fe_index=i) for i in range(len(fes))]

# Now create separate similarity matrices:
# LBP vs LBP (trivial, 1x1)
sim_lbp = cosine_similarity(lbp_vector, lbp_vector)

# Disposable FEs similarity matrix
num_fes = len(disposable_vectors)
similarity_matrix = np.zeros((num_fes, num_fes))
for i in range(num_fes):
    for j in range(num_fes):
        similarity_matrix[i, j] = cosine_similarity(disposable_vectors[i], disposable_vectors[j])

print(f"\n📊 LBP self-similarity: {sim_lbp:.4f}\n")
print("📊 Disposable Feature Extractors similarity matrix:\n")
print(similarity_matrix)

# Plot disposable FEs similarity heatmap
plt.figure(figsize=(12, 10))
sns.set(font_scale=1.0)
sns.set_style("whitegrid")
labels = [f"FE {i}" for i in range(num_fes)]

ax = sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor='gray',
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
    annot_kws={"size": 7},
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Cosine Similarity Between Disposable Feature Extractors", fontsize=16, fontweight='bold')
plt.xlabel("Feature Extractor", fontsize=12)
plt.ylabel("Feature Extractor", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("similarity_heatmap_disposable.png", dpi=300, bbox_inches='tight')
plt.show()
