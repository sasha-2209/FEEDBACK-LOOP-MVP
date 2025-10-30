# classifier.py
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Gemini client
import google.generativeai as genai
# --- We have removed the HttpOptions import ---

load_dotenv()

# --- REVERTED CONFIGURE ---
# We are back to the simple configure call
# This may cause 404 errors again
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
) 

# --- REVERTED MODEL NAME ---
# Try the model name with the "models/" prefix
# as this was required by older library versions
MODEL = "models/gemini-flash-latest"

# ... (rest of the file is unchanged) ...


# System prompt: instruct model to SUMMARIZE a group of texts
GEMINI_SUMMARY_PROMPT = """
You are an expert Product Feedback Intelligence System.
I have a group of customer feedback items that have already been clustered by semantic similarity.
Your job is to analyze all of them and return a single JSON object that summarizes the ENTIRE group.

- **cluster_label**: A single, concise group name (e.g., "Ruby SDK Support", "Billing Invoice Errors").
- **category**: The best fit: <Bug|Feature Request|UX Issue|Performance|SDK Coverage|Billing|Other>
- **priority_score**: An integer (1-5) for the whole cluster's urgency.
- **reasoning**: A one-line summary of the core request or problem.
- **issue_keys**: An array of any Jira keys (e.g., "SDK-123") found in the texts.

Here is the group of feedback items:
{feedback_items_list}

Return ONLY the single JSON object, nothing else.
"""

def get_summary_for_group(texts, model_name=MODEL, max_retries=2, sleep_between_retries=2.0):
    """
    Calls Gemini with the summary prompt for a single group of texts.
    """
    # Format the texts as a bulleted list for the prompt
    prompt_items = "\n".join([f"- \"{t.replace('"',"'").strip()}\"" for t in texts])
    
    batch_prompt = GEMINI_SUMMARY_PROMPT.format(feedback_items_list=prompt_items)

    # Initialize raw to None to prevent UnboundLocalError
    raw = None 

    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(batch_prompt)
            
            raw = resp.text if hasattr(resp, "text") else getattr(resp.parts[0], "text", str(resp.parts))
            
            # Try to find first '{' and last '}'
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
            # Now 'raw' will be None if the API call failed, or the raw text if parsing failed
            print(f"Error parsing group (attempt {attempt+1}): {e}\nRaw output: {raw}")
            time.sleep(sleep_between_retries * (1 + attempt))
            continue
            
    # If we reach here, return an error-filled fallback
    return {
        "cluster_label": "Error: Failed to Summarize",
        "category": "Other",
        "priority_score": 1,
        "reasoning": f"Error classifying batch: {last_err}",
        "issue_keys": []
    }

# Main public function used by app.py
def summarize_clusters(cluster_groups):
    """
    Receives a dict of {cluster_id: [texts]} from the mapper.
    Calls Gemini to summarize each group.
    Returns a consolidated pandas.DataFrame.
    """
    if not cluster_groups:
        return pd.DataFrame()

    agg_rows = []
    
    # Use tqdm for a progress bar in the terminal/console
    for cluster_id, texts in tqdm(cluster_groups.items(), desc="Summarizing clusters with Gemini"):
        if not texts:
            continue

        # Get the summary object from Gemini
        summary = get_summary_for_group(texts)
        
        # Combine with cluster data
        summary["request_count"] = len(texts)
        summary["feedback_text"] = " | ".join(texts) # Join original texts for reference
        
        # Ensure all keys exist
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
    # Filter to only columns that exist, in the right order
    final_cols = [c for c in cols if c in consolidated_df.columns]
    
    consolidated_df = consolidated_df[final_cols] 

    # Sort by priority_score desc then request_count desc
    consolidated_df = consolidated_df.sort_values(
        by=["priority_score", "request_count"], 
        ascending=[False, False]
    ).reset_index(drop=True)

    return consolidated_df