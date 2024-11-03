import requests

# The service URL from your deployment
SERVICE_URL = "https://rag-system-534297186371.us-central1.run.app"

def test_service():
    # Test root endpoint
    response = requests.get(f"{SERVICE_URL}/")
    print("Root endpoint response:", response.json())
    
    # Test health endpoint
    health_response = requests.get(f"{SERVICE_URL}/health")
    print("\nHealth check response:", health_response.json())

if __name__ == "__main__":
    test_service()
