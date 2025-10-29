import requests
import pandas as pd

def fetch_slack_feedback(slack_token, channel_id, limit=200):
    """
    Fetch messages from a Slack channel using Slack API and convert them to a DataFrame.

    Args:
        slack_token (str): Slack Bot User OAuth Token (starts with 'xoxb-')
        channel_id (str): Slack channel ID to fetch messages from
        limit (int): Number of messages to fetch (default: 200)

    Returns:
        pd.DataFrame: DataFrame with columns ['user', 'text', 'ts']
    """
    url = "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {slack_token}"}
    params = {"channel": channel_id, "limit": limit}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Slack API Error: {response.status_code}, {response.text}")
        return pd.DataFrame(columns=["user", "Feedback", "timestamp"])

    data = response.json()
    if not data.get("ok"):
        print(f"Slack API error response: {data}")
        return pd.DataFrame(columns=["user", "Feedback", "timestamp"])

    messages = data.get("messages", [])
    feedback_list = []

    for msg in messages:
        text = msg.get("text", "").strip()
        if text:  # filter out empty messages
            feedback_list.append({
                "user": msg.get("user", "unknown"),
                "Feedback": text,
                "timestamp": msg.get("ts")
            })

    df = pd.DataFrame(feedback_list)
    print(f"✅ Fetched {len(df)} messages from Slack channel {channel_id}")
    return df
