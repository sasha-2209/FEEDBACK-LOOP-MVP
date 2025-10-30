import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# -------------------------
# Config / tuning params
# -------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
# This is the most important setting to tune.
# Lower (e.g., 0.3) = more, smaller, tighter clusters
# Higher (e.g., 0.45) = fewer, larger, looser clusters
DISTANCE_THRESHOLD = 0.35

# -------------------------
# Utilities
# -------------------------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s\+\-#_.]', ' ', t)   # allow + - # . _ as they appear in names
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# -------------------------
# Main function (called by app.py)
# -------------------------
def get_semantic_clusters(feedback_df, text_column):
    """
    Uses sentence embeddings and AgglomerativeClustering to group
    feedback items by semantic similarity.

    Returns:
      A dict mapping {cluster_id: [list_of_original_texts]}
    """

    if feedback_df is None or feedback_df.empty:
        raise ValueError("Feedback DataFrame is empty.")

    if text_column not in feedback_df.columns:
        raise ValueError(f"Selected column '{text_column}' not found in feedback file.")

    # Prepare texts
    original_texts = feedback_df[text_column].astype(str).fillna("").tolist()
    cleaned_texts = [clean_text(t) for t in original_texts]

    if not any(cleaned_texts):
        raise ValueError("No textual feedback found in the selected column.")

    # Step 1: Get embeddings
    model = SentenceTransformer(EMBED_MODEL)
    # Use cleaned texts for embedding, but keep original texts for output
    embeddings = model.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=True)

    # Step 2: Perform clustering
    # If dataset small, create trivial single cluster fallback
    if len(embeddings) == 1:
        initial_labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average"
        )
        initial_labels = clustering.fit_predict(embeddings)

    # Step 3: Build the groups
    clusters = defaultdict(list)
    for i, lbl in enumerate(initial_labels):
        clusters[lbl].append(original_texts[i])

    # Filter out empty strings that may have been clustered
    final_clusters = {}
    for cid, texts in clusters.items():
        valid_texts = [t for t in texts if t and t.strip()]
        if valid_texts:
            final_clusters[cid] = valid_texts
            
    return final_clusters