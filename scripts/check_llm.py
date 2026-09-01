
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import google.auth

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

if not PROJECT_ID:
    raise SystemExit("GCP_PROJECT_ID is not set. See .env.example.")

print(f"Testing Vertex AI Access for {PROJECT_ID}...")

try:
    # Force quota project
    credentials, _ = google.auth.default()
    if hasattr(credentials, 'with_quota_project'):
        credentials = credentials.with_quota_project(PROJECT_ID)

    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    
    model = GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Hello! Are you working?")
    
    print("SUCCESS!")
    print(f"Response: {response.text}")

except Exception as e:
    print(f"FAILURE: {e}")
