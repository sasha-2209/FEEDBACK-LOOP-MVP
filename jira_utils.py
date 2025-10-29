import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
CLOUD_SITE = os.getenv("JIRA_SITE")  # e.g., https://browserstack.atlassian.net
if not CLOUD_SITE:
    raise ValueError("JIRA_SITE is not defined in your .env. Example: https://browserstack.atlassian.net")


def fetch_jira_issues(filter_id, max_results=50):
    """
    Fetch issues from Jira using filter ID and API token auth.
    """
    url = f"{CLOUD_SITE}/rest/api/3/search"
    jql = f"filter={filter_id}"
    params = {"jql": jql, "maxResults": max_results}

    response = requests.get(url, params=params, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    response.raise_for_status()
    return response.json()
