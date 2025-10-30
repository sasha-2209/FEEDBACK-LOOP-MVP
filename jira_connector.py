import os
import requests
import pandas as pd
from dotenv import load_dotenv
import streamlit as st  # <-- ADD THIS IMPORT

load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# --- THIS IS THE FIX ---
# Cache the JIRA API call. If the JQL query is the same,
# Streamlit will return the saved DataFrame instead of hitting the API.
@st.cache_data
def fetch_jira_issues(jql_query):
    """Fetch Jira issues using the new /rest/api/3/search/jql endpoint (fixed)"""
    if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
        raise ValueError("Missing Jira environment variables. Please set JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN.")

    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)

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
            # Return an empty DataFrame instead of raising an error
            return pd.DataFrame(columns=[
                "Issue Key", "Summary", "Description", "Status", 
                "Reporter", "Priority", "ARR", "Deal Size"
            ])

        return pd.DataFrame(issues)

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Error fetching Jira issues: Jira API call failed - {response.status_code} {response.reason}\n"
            f"Details: {response.text}\nPayload: {payload}"
        )
    except Exception as e:
        raise RuntimeError(f"Error fetching Jira issues: {e}")