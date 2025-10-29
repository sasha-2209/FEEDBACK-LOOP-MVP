import requests

def fetch_slack_messages(token, channel, limit=100):
    if not token or not channel:
        raise ValueError("Slack token and channel ID required.")
    url = "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"channel": channel, "limit": limit}
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    if not data.get("ok"):
        raise ValueError(f"Slack API error: {data.get('error')}")
    return [msg["text"] for msg in data.get("messages", []) if "text" in msg]
