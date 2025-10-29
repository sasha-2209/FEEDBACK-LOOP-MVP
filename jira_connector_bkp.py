import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

def fetch_jira_issues(jql_query):
    """Fetch Jira issues using the new /rest/api/3/search/jql endpoint (fixed)"""
    if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
        raise ValueError("Missing Jira environment variables. Please set JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN.")

    # ✅ Correct endpoint for Atlassian's latest Jira Cloud REST API
    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)

    # ✅ Corrected key: use "jql" (not "query")
    payload = {
        "jql": jql_query,
        "maxResults": 200,
        "fields": [
            "summary", 
            "description", 
            "reporter", 
            "status", 
            "priority",
            "customfield_10016",  # ARR (adjust if your field ID differs)
            "customfield_10015"   # Deal size
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, auth=auth)
        response.raise_for_status()
        data = response.json()

        if "issues" not in data:
            raise ValueError(f"Unexpected Jira response format: {data}")

        issues = []
        for issue in data["issues"]:
            fields = issue.get("fields", {})
            issues.append({
                "Issue Key": issue["key"],
                "Summary": fields.get("summary"),
                "Description": fields.get("description"),
                "Status": fields.get("status", {}).get("name"),
                "Reporter": fields.get("reporter", {}).get("displayName"),
                "Priority": fields.get("priority", {}).get("name"),
                "ARR": fields.get("customfield_10016"),
                "Deal Size": fields.get("customfield_10015"),
            })

        if not issues:
            raise ValueError("No issues returned. Check JQL syntax or field permissions.")

        return pd.DataFrame(issues)

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Error fetching Jira issues: Jira API call failed - {response.status_code} {response.reason}\n"
            f"Details: {response.text}\nPayload: {payload}"
        )
    except Exception as e:
        raise RuntimeError(f"Error fetching Jira issues: {e}")
