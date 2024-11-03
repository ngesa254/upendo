
import requests
import json

SERVICE_URL = "https://rag-system-534297186371.us-central1.run.app"

def test_rag():
    # Test chat endpoint
    response = requests.post(
        f"{SERVICE_URL}/chat",
        json={
            "question": "What are the main findings of the report?",
            "session_id": "test-session-1"
        }
    )
    
    print("\nChat Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_rag()
