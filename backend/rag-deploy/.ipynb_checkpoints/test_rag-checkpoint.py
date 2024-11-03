# import requests
# import time
# import logging
# import sys
# from typing import Dict, Any

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class RAGSystemTester:
#     def __init__(self, base_url: str):
#         self.base_url = base_url.rstrip('/')
#         self.session = requests.Session()
#         self.timeout = 30
    
#     def test_health(self) -> bool:
#         """Test the health endpoint"""
#         try:
#             logger.info("Testing health endpoint...")
#             response = self.session.get(
#                 f"{self.base_url}/health",
#                 timeout=self.timeout
#             )
#             data = response.json()
#             logger.info(f"Health Status: {data}")
#             return response.status_code == 200
#         except Exception as e:
#             logger.error(f"Health check failed: {str(e)}")
#             return False

#     def test_chat(self) -> bool:
#         """Test the chat endpoint with various questions"""
#         test_questions = [
#             {
#                 "question": "What are the main findings of the report?",
#                 "session_id": "test-1"
#             },
#             {
#                 "question": "How many developers are there in Africa?",
#                 "session_id": "test-2"
#             }
#         ]

#         for payload in test_questions:
#             try:
#                 logger.info(f"\nTesting chat with question: {payload['question']}")
#                 response = self.session.post(
#                     f"{self.base_url}/chat",
#                     json=payload,
#                     headers={'Content-Type': 'application/json'},
#                     timeout=self.timeout
#                 )
                
#                 logger.info(f"Status Code: {response.status_code}")
#                 if response.status_code == 200:
#                     data = response.json()
#                     logger.info("Answer received successfully")
#                 else:
#                     logger.error(f"Error Response: {response.text}")
#                     return False
                    
#             except requests.exceptions.Timeout:
#                 logger.error("Request timed out. The model might need more time to process.")
#                 return False
#             except Exception as e:
#                 logger.error(f"Chat test failed: {str(e)}")
#                 return False
                
#         return True

#     def run_all_tests(self) -> bool:
#         """Run all tests"""
#         tests_passed = True
        
#         # Test health endpoint
#         if not self.test_health():
#             tests_passed = False
#             logger.error("❌ Health check failed")
#         else:
#             logger.info("✅ Health check passed")

#         # Wait for system initialization
#         logger.info("\nWaiting for system initialization...")
#         time.sleep(10)
        
#         # Test chat endpoint
#         if not self.test_chat():
#             tests_passed = False
#             logger.error("❌ Chat endpoint tests failed")
#         else:
#             logger.info("✅ Chat endpoint tests passed")
            
#         return tests_passed

# def main():
#     SERVICE_URL = "https://rag-system-534297186371.us-central1.run.app"
#     tester = RAGSystemTester(SERVICE_URL)
    
#     logger.info(f"Starting tests for {SERVICE_URL}")
#     success = tester.run_all_tests()
    
#     if success:
#         logger.info("\n✅ All tests passed successfully!")
#         sys.exit(0)
#     else:
#         logger.error("\n❌ Some tests failed!")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()



import requests
import time
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = 30  # 30 second timeout
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            logger.info("Testing health endpoint...")
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def test_chat(self) -> bool:
        """Test chat endpoint"""
        test_questions = [
            {
                "question": "What are the main findings of the report?",
                "session_id": "test-session-1"
            },
            {
                "question": "How many developers are in Africa?",
                "session_id": "test-session-2"
            }
        ]

        for question in test_questions:
            try:
                logger.info(f"\nTesting chat with question: {question['question']}")
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=question,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.timeout
                )
                
                logger.info(f"Status Code: {response.status_code}")
                logger.info(f"Response Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Answer: {data.get('answer', '')[:200]}...")
                    logger.info(f"Number of sources: {len(data.get('sources', []))}")
                else:
                    logger.error(f"Error Response: {response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                logger.error("Request timed out. The model might need more time to process.")
                return False
            except Exception as e:
                logger.error(f"Chat test failed: {str(e)}")
                return False
                
        return True

def main():
    SERVICE_URL = "https://rag-system-534297186371.us-central1.run.app"
    tester = RAGTester(SERVICE_URL)
    
    # Test health endpoint
    if not tester.test_health():
        logger.error("❌ Health check failed")
        return
    logger.info("✅ Health check passed")
    
    # Wait for system initialization
    logger.info("\nWaiting for system initialization...")
    time.sleep(10)
    
    # Test chat endpoint
    if not tester.test_chat():
        logger.error("❌ Chat endpoint tests failed")
        return
    logger.info("✅ Chat endpoint tests passed")
    
    logger.info("\n✅ All tests passed successfully!")

if __name__ == "__main__":
    main()