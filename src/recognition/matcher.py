import pickle
import numpy as np
from src.utils import cosine_similarity


def FaceMatcher(
    query_embedding,
    embeddings_file_path=r"D:\Face_Attendence\data\embeddings.pkl",
    top_k=5
):
    # Load stored embeddings
    with open(embeddings_file_path, "rb") as f:
        all_embeddings = pickle.load(f)

    scores = {}

    # Loop over each registered user
    for user_id, embeddings in all_embeddings.items():
        similarities = []

        # Compare query with each stored embedding
        for emb in embeddings:
            sim = cosine_similarity(query_embedding, emb)
            similarities.append(sim)

        # Sort similarities: highest first (cosine similarity)
        similarities = sorted(similarities, reverse=True)

        # Take top-K closest embeddings
        top_k_similarities = similarities[:top_k]

        # Average score
        avg_score = sum(top_k_similarities) / len(top_k_similarities)

        scores[user_id] = avg_score

    # Pick the user with highest similarity
    best_match = max(scores, key=scores.get)

    return best_match
