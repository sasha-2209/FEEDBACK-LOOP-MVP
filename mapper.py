import re
import ast
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from sklearn.cluster import AgglomerativeClustering

# -------------------------
# Config / tuning params
# -------------------------
EMBED_MODEL = "local_model"
EMBED_MODEL_PATH = os.path.abspath(EMBED_MODEL)

# Load the model ONCE when the script is imported.
try:
    MODEL = SentenceTransformer(EMBED_MODEL_PATH)
except Exception as e:
    print(f"!!!!!!!!!!!!!! FAILED TO LOAD MODEL !!!!!!!!!!!!!!")
    print(f"Error: {e}")
    print(f"Please check that the 'local_model' folder is not empty and contains all model files.")
    raise e

DISTANCE_THRESHOLD = 0.35
SIMILARITY_THRESHOLD = 0.60

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
# Step 3 Main function (called by app.py)
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
    embeddings = MODEL.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=True)

    # Step 2: Perform clustering
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


# -----------------------------------------------------------------
# --- FUNCTION FOR STEP 4 ---
# -----------------------------------------------------------------
def map_feedback_to_dealblockers(feedback_df, jira_df):
    """
    Maps consolidated feedback clusters to Jira dealblockers using a
    hybrid approach:
    1. Explicit key matching (if feedback contains a Jira key)
    2. Semantic similarity matching (if no key is found)
    """
    if feedback_df.empty or jira_df.empty:
        return pd.DataFrame()

    all_mappings = []
    
    # --- Setup Jira side ---
    jira_key_set = set(jira_df['Issue Key'])
    jira_summaries = jira_df['Summary'].fillna('').astype(str).tolist()
    jira_embeddings = MODEL.encode(jira_summaries, normalize_embeddings=True, show_progress_bar=True)
    
    unmatched_feedback_rows = []

    # --- Pass 1: Explicit Key Matching ---
    for _, fb_row in feedback_df.iterrows():
        try:
            issue_keys = ast.literal_eval(str(fb_row['issue_keys']))
        except (ValueError, SyntaxError):
            issue_keys = []
        
        explicitly_matched = False
        for key in issue_keys:
            if key in jira_key_set:
                jira_row = jira_df[jira_df['Issue Key'] == key].iloc[0]
                
                all_mappings.append({
                    "cluster_label": fb_row['cluster_label'],
                    "feedback_reasoning": fb_row['reasoning'],
                    "request_count": fb_row['request_count'],
                    "mapped_issue_key": jira_row['Issue Key'],
                    "mapped_issue_summary": jira_row['Summary'],
                    "match_type": "Explicit Key",
                    "match_score": 1.0
                })
                explicitly_matched = True

        if not explicitly_matched:
            unmatched_feedback_rows.append(fb_row)

    # --- Pass 2: Semantic Similarity Matching ---
    if unmatched_feedback_rows:
        unmatched_df = pd.DataFrame(unmatched_feedback_rows)
        
        feedback_texts = unmatched_df['reasoning'].fillna(unmatched_df['cluster_label']).astype(str).tolist()
        
        if feedback_texts:
            feedback_embeddings = MODEL.encode(feedback_texts, normalize_embeddings=True, show_progress_bar=True)
            
            cos_scores = pytorch_cos_sim(feedback_embeddings, jira_embeddings)
            
            # Find the best match for each feedback item
            for i, fb_row in enumerate(unmatched_df.itertuples()):
                
                # --- THIS IS THE FIX ---
                # .argmax() returns a tensor, so we add .item() to convert it to a Python integer
                best_match_idx = cos_scores[i].argmax().item()
                best_score = cos_scores[i][best_match_idx].item() # .item() was already here, which is good
                # -----------------------
                
                if best_score >= SIMILARITY_THRESHOLD:
                    # Now 'best_match_idx' is an integer, so .iloc will work
                    jira_row = jira_df.iloc[best_match_idx]
                    
                    all_mappings.append({
                        "cluster_label": fb_row.cluster_label,
                        "feedback_reasoning": fb_row.reasoning,
                        "request_count": fb_row.request_count,
                        "mapped_issue_key": jira_row['Issue Key'],
                        "mapped_issue_summary": jira_row['Summary'],
                        "match_type": "Semantic Match",
                        "match_score": best_score
                    })

    # --- Finalize ---
    if not all_mappings:
        return pd.DataFrame()
        
    final_df = pd.DataFrame(all_mappings)
    final_df = final_df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
    return final_df