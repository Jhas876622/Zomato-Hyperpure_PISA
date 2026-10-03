# =============================================================
# alerts.py — Push alerts to a chat webhook
#
# Set ALERT_WEBHOOK_URL to a Slack, Discord or Microsoft Teams incoming-webhook
# URL. The payload carries both "text" (Slack/Teams) and "content" (Discord).
# Without the variable, alerts are only printed to the log.
# =============================================================

import os

import requests

ICONS = {"error": "🔴", "warning": "🟠", "info": "🟢"}


def send_alert(title, details=(), level="error"):
    """Best effort: never raises, so a broken webhook can't fail the pipeline. Returns True if delivered."""
    text = f"{ICONS.get(level, '')} PISA {level.upper()}: {title}" + "".join(f"\n• {d}" for d in details)
    print(text)
    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        return False
    try:
        resp = requests.post(url, json={"text": text, "content": text}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"   ⚠️  Alert webhook failed: {e}")
        return False
