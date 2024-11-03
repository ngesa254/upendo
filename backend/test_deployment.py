import requests

# Get your service URL after deployment
SERVICE_URL = "YOUR_CLOUD_RUN_URL"  # Replace with actual URL

# https://rag-system-534297186371.us-central1.run.app

# Test health endpoint
health_response = requests.get(f"{SERVICE_URL}/health")
print("Health check:", health_response.json())

# Test chat endpoint
chat_response = requests.post(
    f"{SERVICE_URL}/chat",
    json={
        "question": "What are the main findings of the report?",
        "session_id": "test-1"
    }
)
print("\nChat Response:", chat_response.json())