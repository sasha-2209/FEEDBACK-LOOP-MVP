import re
import ast
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from sklearn.cluster import AgglomerativeClustering
import streamlit as st  # <-- ADD THIS IMPORT

# -------------------------
# Config / tuning params
# -------------------------

# --- THIS IS THE FIX ---
# We cache the model load, so it only runs ONCE.
@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model into Streamlit's cache."""
    EMBED_MODEL = "local_model"
    EMBED_MODEL_PATH = os.path.abspath(EMBED_MODEL)
    try:
        model = SentenceTransformer(EMBED_MODEL_PATH)
        return model
    except Exception as e:
        print(f"!!!!!!!!!!!!!! FAILED TO LOAD MODEL !!!!!!!!!!!!!!")
        print(f"Error: {e}")
        st.error(f"Error loading embedding model from {EMBED_MODEL_PATH}. Check folder exists.")
        return None
# ---------------------

DISTANCE_THRESHOLD = 0.35
#SIMILARITY_THRESHOLD = 0.60

# -------------------------
# Utilities
# -------------------------
def clean_text(t):
    # ... (function is unchanged) ...
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s\+\-#_.]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# -------------------------
# Step 3 Main function (called by app.py)
# -------------------------
# We cache the clustering result. If the input df is the same,
# it will return the cached groups instantly.
@st.cache_data
def get_semantic_clusters(feedback_df, text_column, grouping_context=""):
    """
    Uses sentence embeddings and AgglomerativeClustering to group
    feedback items by semantic similarity.
    """
    MODEL = load_embedding_model() 
    if MODEL is None:
        st.error("Model not loaded. Halting clustering.")
        st.stop()

    if feedback_df is None or feedback_df.empty:
        raise ValueError("Feedback DataFrame is empty.")

    if text_column not in feedback_df.columns:
        raise ValueError(f"Selected column '{text_column}' not found in feedback file.")

    # Prepare texts
    original_texts = feedback_df[text_column].astype(str).fillna("").tolist()
    
    # --- 2. ADD THIS LOGIC TO PREPEND CONTEXT ---
    cleaned_texts = [clean_text(t) for t in original_texts]
    if grouping_context and grouping_context.strip():
        # If context is provided, prepend it to every item
        # This will influence the vector math and change the groups
        clean_context = clean_text(grouping_context)
        cleaned_texts = [f"Context: {clean_context}. Feedback: {t}" for t in cleaned_texts]
    # ----------------------------------------------

    if not any(cleaned_texts):
        raise ValueError("No textual feedback found in the selected column.")

    # Step 1: Get embeddings
    embeddings = MODEL.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=True)

    # Step 2: Perform clustering
    # ... (rest of the function is unchanged) ...
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
        # We still return the ORIGINAL text, not the one with context
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
# We also cache the mapping step.
@st.cache_data
def map_feedback_to_dealblockers(feedback_df, jira_df, similarity_threshold=0.7):
    """
    Maps consolidated feedback clusters to Jira dealblockers using a
    hybrid approach.
    """
    MODEL = load_embedding_model() # Get the cached model
    if MODEL is None:
        st.error("Model not loaded. Halting mapping.")
        st.stop()
        
    if feedback_df.empty or jira_df.empty:
        return pd.DataFrame()

    all_mappings = []
    
    # --- Setup Jira side ---
    jira_key_set = set(jira_df['Issue Key'])
    jira_summaries = jira_df['Summary'].fillna('').astype(str).tolist()
    jira_embeddings = MODEL.encode(jira_summaries, normalize_embeddings=True, show_progress_bar=True)
    
    unmatched_feedback_rows = []

    # --- Pass 1: Explicit Key Matching ---
    # --- Pass 1: Explicit Key Matching ---
    # --- Pass 1: Explicit Key Matching ---
    for _, fb_row in feedback_df.iterrows():

        # --- START FIX ---
        # Robustly parse the 'issue_keys' column
        issue_keys_str = str(fb_row.get('issue_keys', '[]'))
        
        # Handle cases where the value is None, NaN, or an empty string
        if issue_keys_str.lower() in ('', 'nan', 'none', 'null'):
            issue_keys = []
        else:
            try:
                # Try to evaluate the string as a Python literal
                issue_keys = ast.literal_eval(issue_keys_str)
                # Ensure the result is actually a list
                if not isinstance(issue_keys, list):
                    issue_keys = []
            except (ValueError, SyntaxError):
                # Fail safely to an empty list
                issue_keys = []
        # --- END FIX ---

        explicitly_matched = False
        for key in issue_keys:
            if key in jira_key_set:
                jira_row = jira_df[jira_df['Issue Key'] == key].iloc[0]
                
                all_mappings.append({
                    "cluster_label": fb_row['cluster_label'],
                    "feedback_reasoning": fb_row['reasoning'],
                    "request_count": fb_row['request_count'],
                    "original_feedback_texts": fb_row['feedback_text'],
                    "extracted_feedback_keys": fb_row['issue_keys'],
                    "mapped_issue_key": jira_row['Issue Key'],
                    "mapped_issue_summary": jira_row['Summary'],
                    "match_type": "Explicit Key",
                    "match_score": 1.0
                })
                explicitly_matched = True

        if not explicitly_matched:
            unmatched_feedback_rows.append(fb_row)

    # --- Pass 2: Semantic Similarity Matching ---

    # --- Pass 2: Semantic Similarity Matching ---
    if unmatched_feedback_rows:
        unmatched_df = pd.DataFrame(unmatched_feedback_rows)
        
        feedback_texts = unmatched_df['reasoning'].fillna(unmatched_df['cluster_label']).astype(str).tolist()
        
        if feedback_texts:
            feedback_embeddings = MODEL.encode(feedback_texts, normalize_embeddings=True, show_progress_bar=True)
            
            cos_scores = pytorch_cos_sim(feedback_embeddings, jira_embeddings)
            
            # Find the best match for each feedback item
            for i, fb_row in enumerate(unmatched_df.itertuples()):
                
                best_match_idx = cos_scores[i].argmax().item()
                best_score = cos_scores[i][best_match_idx].item()
                
                if best_score >= similarity_threshold:
                    jira_row = jira_df.iloc[best_match_idx]
                    
                    all_mappings.append({
                        "cluster_label": fb_row.cluster_label,
                        "feedback_reasoning": fb_row.reasoning,
                        "request_count": fb_row.request_count,
                        "original_feedback_texts": fb_row.feedback_text,
                        "extracted_feedback_keys": fb_row.issue_keys,
                        "mapped_issue_key": jira_row['Issue Key'],
                        "mapped_issue_summary": jira_row['Summary'],
                        "match_type": "Semantic Match",
                        "match_score": best_score
                    })

    # --- Finalize ---
    if not all_mappings:
        return pd.DataFrame()
        
    final_df = pd.DataFrame(all_mappings)
    
    # Re-order columns to be more logical
    final_cols = [
        "cluster_label", "feedback_reasoning", "request_count",
        "mapped_issue_key", "mapped_issue_summary", "match_type", "match_score",
        "original_feedback_texts", "extracted_feedback_keys"
    ]
    final_df = final_df[[c for c in final_cols if c in final_df.columns]]
    
    final_df = final_df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
    return final_df