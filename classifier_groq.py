import os
import json
import time
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a Product Feedback Intelligence Assistant.
You receive multiple lines of user feedback. 
Your task is to analyze all of them together and produce a JSON summary that helps a Product Manager prioritize work.

Follow these rules:
1. Group similar feedback items into **requests** (e.g. “billing confusion” and “invoice errors” → one group).
2. For each unique request, output:
   - request: the concise ask or problem
   - category: one-word theme (Feature, Bug, UI/UX, Performance, New Framework Coverage, etc.)
   - related_feedback_keys: list of Jira issue keys or feedback references that belong to this group
   - request_count: number of feedback items in this group
   - urgency: High / Medium / Low — based on how time-sensitive or user-impacting the issue is
   - priority: High / Medium / Low — overall business priority considering frequency + urgency + dollar impact
   - explanation: one line explaining why you rated this request with that priority and urgency

3. After grouping, sort the JSON output by **priority** (High → Medium → Low), and then by **request_count** (descending).

Return only valid JSON array of objects, e.g.:

[
  {
    "request": "Fix billing confusion and invoice errors",
    "category": "Pricing",
    "related_feedback_keys": ["PRDFBK-100", "PRDFBK-112", "PRDFBK-119"],
    "request_count": 3,
    "urgency": "High",
    "priority": "High",
    "explanation": "Frequent billing complaints with monetary impact; immediate business value."
  },
  ...
]
"""


EXAMPLE_PROMPT = """
Example:
Input: "We're spending a lot because your billing is confusing and invoices are wrong."
Output:
{"category":"Pricing","request":"Fix billing/invoicing errors","similar_requests":["Billing issues"],"request_count":7,"dollar_impact":"High","urgency":"High","sentiment":"Negative","priority":"High","explanation":"Mentions billing errors and monetary impact, high urgency."}
"""

def analyze_texts_batch(texts):
    results = []
    for text in tqdm(texts, desc="Classifying feedback"):
        try:
            prompt = f"{SYSTEM_PROMPT}\n{EXAMPLE_PROMPT}\nInput: \"{text}\"\nOutput:"
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # fast + accurate
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            result_text = resp.choices[0].message.content.strip()

            # Extract JSON
            start, end = result_text.find("{"), result_text.rfind("}") + 1
            if start != -1 and end != -1:
                json_text = result_text[start:end]
                parsed = json.loads(json_text)
                results.append(parsed)
            else:
                results.append({
                    "category": "Other",
                    "request": text[:80],
                    "similar_requests": [],
                    "request_count": 1,
                    "dollar_impact": "Low",
                    "urgency": "Low",
                    "sentiment": "Neutral",
                    "priority": "Low",
                    "explanation": "Could not parse model output"
                })
            time.sleep(0.1)
        except Exception as e:
            results.append({
                "category": "Other",
                "request": text[:80],
                "similar_requests": [],
                "request_count": 1,
                "dollar_impact": "Low",
                "urgency": "Low",
                "sentiment": "Neutral",
                "priority": "Low",
                "explanation": f"Error: {e}"
            })
    return results
