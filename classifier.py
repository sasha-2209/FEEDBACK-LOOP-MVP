import os
import json
import math
import re
from tqdm import tqdm
import google.generativeai as genai
import pandas as pd  # ✅ Added missing import

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "models/gemini-2.5-pro"

system_prompt = """
You are an expert Product Feedback Intelligence System built for Product Managers.
Your job is to turn raw, messy user feedback into structured, actionable insights
that help prioritize roadmap items.

You are deeply analytical, concise, and explain your reasoning clearly.

🔹 Your output should be a JSON array where each object contains:
- "feedback_text": Original feedback snippet
- "category": Bug / Feature Request / UX Issue / Performance / SDK Coverage - Depth | SDK Coverage - Breadth
- "cluster_label": Cluster name that groups similar feedback
- "request_count": number of similar mentions
- "issue_keys": list of sample issue identifiers (if available in the dataset)
- "priority_score": Integer (1–5), where 5 = most urgent, most frequent & impactful
- "reasoning": Brief reasoning for category + priority

If no explicit issue identifiers are provided, use "issue_keys": ["AUTO-GEN-{index}"] placeholders.
"""

def chunk_list(lst, n):
    """Split list into n-sized chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def analyze_texts_batch(feedback_items, issue_keys=None, batch_size=20):
    """Analyze batches of feedback using Gemini 2.5 Pro"""
    model = genai.GenerativeModel(model_name=MODEL)
    results = []

    if not feedback_items:
        return []

    # Combine text + issue key context
    combined_feedback = []
    for i, text in enumerate(feedback_items):
        key_info = ""
        if issue_keys and i < len(issue_keys) and not pd.isna(issue_keys[i]):
            key_info = f"Issue Key: {issue_keys[i]}"
        else:
            key_info = f"Issue Key: AUTO-GEN-{i+1}"
        combined_feedback.append(f"{key_info}\nFeedback: {text}")

    # Process in batches
    for batch_index, batch in enumerate(tqdm(list(chunk_list(combined_feedback, batch_size)))):
        joined_text = "\n\n---\n\n".join(batch)
        prompt = f"{system_prompt}\n\nAnalyze the following feedback entries:\n\n{joined_text}"

        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text"):
                results.append(response.text)
            else:
                results.append("Error: No text in response")
        except Exception as e:
            results.append(f"Error in batch {batch_index}: {e}")
            continue

    return results
