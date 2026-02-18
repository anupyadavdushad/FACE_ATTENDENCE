import numpy as np
from config.settings import EMBEDDINGS_FILE, ATTENDANCE_LOG_DIR
from src.utils import load_pickle, ensure_dir, save_attendance_csv
from src.recognition.embedder import FaceEmbedder


def match_face(frame, threshold=0.6):
    """
    Returns: (status, best_match, score)

    status = "MATCH" or "UNKNOWN" or "NO_FACE"
    best_match = {"name":..., "reg_no":...} or None
    """

    embedder = FaceEmbedder()
    db = load_pickle(EMBEDDINGS_FILE)

    face = embedder.get_face(frame)

    if face is None:
        return "NO_FACE", None, None

    emb = embedder.get_embedding(face)

    best_match = None
    best_score = -1

    for record in db:
        stored_emb = record["embedding"]
        similarity = np.dot(emb, stored_emb)

        if similarity > best_score:
            best_score = similarity
            best_match = record

    if best_match and best_score >= threshold:
        return "MATCH", best_match, best_score

    return "UNKNOWN", None, best_score


def mark_attendance(best_match):
    ensure_dir(ATTENDANCE_LOG_DIR)
    name = best_match["name"]
    reg_no = best_match["reg_no"]

    filepath = save_attendance_csv(ATTENDANCE_LOG_DIR, name, reg_no)
    return filepath
