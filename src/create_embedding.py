import os
import cv2
import numpy as np

from config.settings import RAW_DATA_DIR, EMBEDDINGS_FILE, EMBEDDINGS_DIR
from src.utils import ensure_dir, save_pickle
from src.recognition.embedder import FaceEmbedder


def create_embeddings():
    ensure_dir(EMBEDDINGS_DIR)

    embedder = FaceEmbedder()
    embeddings_db = []

    for student_folder in os.listdir(RAW_DATA_DIR):
        folder_path = os.path.join(RAW_DATA_DIR, student_folder)

        if not os.path.isdir(folder_path):
            continue

        reg_no, name = student_folder.split("_", 1)

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            face = embedder.get_face(img)
            if face is None:
                continue

            embedding = embedder.get_embedding(face)

            embeddings_db.append({
                "name": name,
                "reg_no": reg_no,
                "embedding": embedding
            })

    save_pickle(embeddings_db, EMBEDDINGS_FILE)
    return EMBEDDINGS_FILE
