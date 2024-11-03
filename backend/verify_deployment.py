import requests
import time
import logging
from typing import Dict, Any, Optional
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceVerifier:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        
    def verify_health(self) -> bool:
        """Verify health endpoint"""
        try:
            response = self.session.get(f"{self.service_url}/health")
            logger.info(f"Health check status: {response.status_code}")
            logger.info(f"Health check response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def verify_chat(self) -> bool:
        """Verify chat endpoint"""
        test_payload = {
            "question": "What is this report about?",
            "session_id": "test-verification"
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/chat",
                json=test_payload,
                headers={'Content-Type': 'application/json'}
            )
            logger.info(f"Chat test status: {response.status_code}")
            logger.info(f"Chat test response: {response.text[:200]}...")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Chat test failed: {str(e)}")
            return False
    
    def verify_all(self) -> bool:
        """Run all verifications"""
        logger.info("Starting service verification...")
        
        # Check health endpoint
        if not self.verify_health():
            logger.error("Health check verification failed")
            return False
            
        # Wait for initialization
        logger.info("Waiting for service initialization...")
        time.sleep(10)
        
        # Check chat endpoint
        if not self.verify_chat():
            logger.error("Chat endpoint verification failed")
            return False
            
        logger.info("All verifications passed successfully!")
        return True

def main():
    service_url = "https://rag-system-534297186371.us-central1.run.app"
    verifier = ServiceVerifier(service_url)
    
    if verifier.verify_all():
        print("\n✅ Service verification completed successfully!")
    else:
        print("\n❌ Service verification failed!")
        exit(1)

if __name__ == "__main__":
    main()