"""
check_grok.py
-------------
Standalone test for your xAI Grok API key -- isolates auth problems from
the rest of the app. Paste your key below and run:

    python check_grok.py

A 200 response with a short reply means the key is good. A 403 means
xAI is rejecting the request before it even gets to the model -- almost
always a missing payment method on the account, a key with a stray
space/newline pasted in, or a revoked/wrong-project key. Check
console.x.ai (Billing and API Keys tabs) if you see one.
"""

import requests

API_KEY = ""  # paste your xai-... key here

resp = requests.post(
    "https://api.x.ai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "grok-4.3",
        "messages": [{"role": "user", "content": "Reply with just the word OK."}],
        "max_tokens": 5,
    },
    timeout=30,
)

print("Status:", resp.status_code)
print("Body:", resp.text[:1000])
