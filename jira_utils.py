import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
CLOUD_SITE = os.getenv("JIRA_SITE")  # e.g., https://browserstack.atian.net
if not CLOUD_SITE:
    raise ValueError("JIRA_SITE is not defined in your .env. Example: https://browserstack.atlassian.net")


def fetch_jira_issues(filter_id, max_results=200):
    """
    Fetch issues from Jira using filter ID and API token auth.
    """
    url = f"{CLOUD_SITE}/rest/api/3/search"
    jql = f"filter={filter_id}"

    # --- START MODIFICATIONS ---

    # We must ensure max_results is an integer.
    # The 'bad argument type' error likely comes from max_results
    # being None or an empty string, which cannot be processed.
    try:
        # Convert the incoming max_results to an integer
        valid_max_results = int(max_results)
    except (ValueError, TypeError):
        # If conversion fails (e.g., it was None, "", or "abc"),
        # fall back to the default value.
        valid_max_results = 200

    # Use the validated integer in your parameters
    params = {"jql": jql, "maxResults": valid_max_results}

    # --- END MODIFICATIONS ---

    response = requests.get(url, params=params, auth=HTTPBasicAuth(EMAIL, API_TOKEN))
    response.raise_for_status()
    return response.json()