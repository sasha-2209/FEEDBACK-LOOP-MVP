from fuzzywuzzy import fuzz
import pandas as pd

def map_feedback_to_dealblockers(feedback_df, jira_df, threshold=60):
    feedback_matches = []
    for _, fb_row in feedback_df.iterrows():
        best_match = None
        best_score = 0

        for _, jira_row in jira_df.iterrows():
            score = fuzz.token_set_ratio(fb_row["feedback_text"], jira_row["Summary"])
            if score > best_score:
                best_score = score
                best_match = jira_row

        if best_match is not None and best_score >= threshold:
            feedback_matches.append({
                "feedback_text": fb_row["feedback_text"],
                "category": fb_row.get("category", ""),
                "cluster_label": fb_row.get("cluster_label", ""),
                "request_count": fb_row.get("request_count", 1),
                "issue_keys": fb_row.get("issue_keys", ""),
                "priority_score": fb_row.get("priority_score", ""),
                "reasoning": fb_row.get("reasoning", ""),
                "matched_issue_key": best_match["Issue Key"],
                "matched_summary": best_match["Summary"],
                "match_score": best_score
            })

    feedback_table = pd.DataFrame(feedback_matches)

    # Create Dealblocker Summary
    dealblocker_summary = []
    for _, row in jira_df.iterrows():
        urgency_score = 3
        reason = "Normal priority"
        desc = str(row["Description"]).lower()

        if any(word in desc for word in ["urgent", "critical", "blocker"]):
            urgency_score = 5
            reason = "Contains urgency markers (urgent/critical/blocker)"
        elif "delay" in desc:
            urgency_score = 4
            reason = "Mentions delay or time sensitivity"

        dealblocker_summary.append({
            "Issue Key": row["Issue Key"],
            "Summary": row["Summary"],
            "Description": row["Description"],
            "Urgency Score": urgency_score,
            "Urgency Reason": reason,
            "ARR Value": row["ARR Value"],
            "Deal Size": row["Deal Size"],
            "Timeline Urgency": row["Timeline Urgency"]
        })

    dealblocker_table = pd.DataFrame(dealblocker_summary)

    return feedback_table, dealblocker_table
