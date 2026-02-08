import httpx
BASE_URL = "http://localhost:8000"
# 1. Get API Key
resp = httpx.post(f"{BASE_URL}/api/v1/keys/", json={"name": "test"})
api_key = resp.json()["api_key"]
print(f"API Key: {api_key}")
# 2. Create Session
resp = httpx.post(
    f"{BASE_URL}/api/v1/documents/sessions",
    headers={"X-API-Key": api_key}
)
session_id = resp.json()["session_id"]
print(f"Session ID: {session_id}")
# 3. Upload (skip if already have documents)
# ...
# 4. Query
resp = httpx.post(
    f"{BASE_URL}/api/v1/query/{session_id}",
    headers={"X-API-Key": api_key},
    json={"question": "What is the main topic?"},
    timeout=60.0
)
print(f"Response: {resp.json()}")