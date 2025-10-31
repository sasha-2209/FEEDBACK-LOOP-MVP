# classifier.py
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import streamlit as st 

# Gemini client
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
) 

MODEL = "models/gemini-flash-latest"


# --- 1. MODIFY THE PROMPT ---
# Added a placeholder {user_context_section}
GEMINI_SUMMARY_PROMPT = """
You are an expert Product Feedback Intelligence System.

Your job is to analyze all of the following feedback items and return a single JSON object that summarizes the ENTIRE group.

- **cluster_label**: A single, concise group name (e.g., "Ruby SDK Support", "Billing Invoice Errors").
- **category**: The best fit: <Bug|Feature Request|UX Issue|Performance|SDK Coverage|Billing|Other>
- **priority_score**: An integer (1-5) for the whole cluster's urgency.
- **reasoning**: A one-line summary of the core request or problem.
- **issue_keys**: An array of any Jira keys (e.g., "SDK-123") found in the texts.
In addition to the above, also keep the following in mind when analyzing the group: {user_context_section}

Here is the group of feedback items:
{feedback_items_list}

Return ONLY the single JSON object, nothing else.
"""

# --- 2. MODIFY THIS FUNCTION SIGNATURE ---
def get_summary_for_group(texts, labeling_context="", model_name=MODEL, max_retries=2, sleep_between_retries=2.0):
    """
    Calls Gemini with the summary prompt for a single group of texts.
    """
    
    # --- 3. ADD THIS LOGIC to dynamically build the prompt ---
    if labeling_context and labeling_context.strip():
        context_section = f"A user has provided this context, please use it to guide your summary: '{labeling_context}'\n"
    else:
        context_section = "" # If no context, this part is empty
        
    prompt_items = "\n".join([f"- \"{t.replace('"',"'").strip()}\"" for t in texts])
    
    batch_prompt = GEMINI_SUMMARY_PROMPT.format(
        user_context_section=context_section,
        feedback_items_list=prompt_items
    )
    # --------------------------------------------------------

    raw = None 
    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(batch_prompt)
            
            raw = resp.text if hasattr(resp, "text") else getattr(resp.parts[0], "text", str(resp.parts))
            
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end != -1:
                json_text = raw[start:end]
            else:
                json_text = raw
                
            parsed = json.loads(json_text)
            return parsed
        
        except Exception as e:
            last_err = e
            print(f"Error parsing group (attempt {attempt+1}): {e}\nRaw output: {raw}")
            time.sleep(sleep_between_retries * (1 + attempt))
            continue
            
    return {
        "cluster_label": "Error: Failed to Summarize",
        "category": "Other",
        "priority_score": 1,
        "reasoning": f"Error classifying batch: {last_err}",
        "issue_keys": []
    }

# --- 4. MODIFY THIS FUNCTION SIGNATURE ---
@st.cache_data
def summarize_clusters(cluster_groups, labeling_context=""):
    """
    Receives a dict of {cluster_id: [texts]} from the mapper.
    Calls Gemini to summarize each group.
    Returns a consolidated pandas.DataFrame.
    """
    if not cluster_groups:
        return pd.DataFrame()

    agg_rows = []
    
    for cluster_id, texts in tqdm(cluster_groups.items(), desc="Summarizing clusters with Gemini"):
        if not texts:
            continue

        # --- 5. PASS THE CONTEXT DOWN ---
        summary = get_summary_for_group(texts, labeling_context=labeling_context)
        
        # Combine with cluster data
        summary["request_count"] = len(texts)
        summary["feedback_text"] = " | ".join(texts) 
        
        summary.setdefault("cluster_label", "Untitled Cluster")
        summary.setdefault("category", "Other")
        summary.setdefault("priority_score", 1)
        summary.setdefault("reasoning", "")
        summary.setdefault("issue_keys", [])

        agg_rows.append(summary)

    consolidated_df = pd.DataFrame(agg_rows)

    # Re-order columns for clarity
    cols = [
        "cluster_label", "category", "priority_score", "request_count", 
        "reasoning", "issue_keys", "feedback_text"
    ]
    final_cols = [c for c in cols if c in consolidated_df.columns]
    
    consolidated_df = consolidated_df[final_cols] 

    consolidated_df = consolidated_df.sort_values(
        by=["priority_score", "request_count"], 
        ascending=[False, False]
    ).reset_index(drop=True)

    return consolidated_df