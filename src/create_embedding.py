import os
import pickle
from recognition.embedder import FaceEmbedder
import cv2
import numpy as np

# -------- CONFIG --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
EMBEDDING_PATH = os.path.join(BASE_DIR, "data", "embeddings.pkl")
# ------------------------

embedder = FaceEmbedder()
all_embeddings = {}

for user_id in os.listdir(RAW_DATA_DIR):
    user_path = os.path.join(RAW_DATA_DIR, user_id)

    if not os.path.isdir(user_path):
        continue

    print(f"[INFO] Processing user: {user_id}")
    embeddings = []

    for img_name in os.listdir(user_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(user_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Failed to read {img_path}")
            continue

        emb = embedder.get_embedding(img)
        embeddings.append(emb)

    if embeddings:
        all_embeddings[user_id] = np.array(embeddings)
        print(f"[INFO] Saved {len(embeddings)} embeddings for {user_id}")
    else:
        print(f"[WARN] No valid images for {user_id}")

with open(EMBEDDING_PATH, "wb") as f:
    pickle.dump(all_embeddings, f)

print(f"[DONE] All embeddings saved to {EMBEDDING_PATH}")
