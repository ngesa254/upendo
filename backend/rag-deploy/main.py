# # import os
# # from typing import List, Dict, Any
# # from dataclasses import dataclass
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores.chroma import Chroma
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.prompts import PromptTemplate
# # from langchain.memory import ConversationBufferMemory

# # class DocumentChat:
# #     def __init__(self, pdf_path: str):
# #         self.pdf_path = pdf_path
# #         self.llm = ChatVertexAI(
# #             model_name="gemini-pro",
# #             temperature=0.7,
# #             max_output_tokens=1024
# #         )
# #         self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
# #         self.memory = ConversationBufferMemory(
# #             memory_key="chat_history",
# #             return_messages=True,
# #             output_key="answer"
# #         )
# #         self.qa_chain = None
        
# #     def initialize_system(self):
# #         """Initialize the document chat system"""
# #         print("Loading document...")
# #         loader = PyPDFLoader(self.pdf_path)
# #         documents = loader.load()
        
# #         print("Processing document...")
# #         text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=1000,
# #             chunk_overlap=100
# #         )
# #         splits = text_splitter.split_documents(documents)
        
# #         print("Creating vector store...")
# #         vectorstore = Chroma.from_documents(
# #             documents=splits,
# #             embedding=self.embeddings
# #         )
        
# #         # Create the condense question prompt
# #         condense_question_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

# # Chat History:
# # {chat_history}
# # Follow Up Input: {question}
# # Standalone question:""")
        
# #         # Create the QA prompt with explicit input variables
# #         qa_prompt = PromptTemplate(
# #             template="""Use the following pieces of context to answer the question at the end.
            
# # Context:
# # {context}

# # Question: {question}

# # Answer:""",
# #             input_variables=["context", "question"]
# #         )
        
# #         # Create the chain with proper configuration
# #         self.qa_chain = ConversationalRetrievalChain.from_llm(
# #             llm=self.llm,
# #             retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
# #             memory=self.memory,
# #             get_chat_history=lambda h: str(h),
# #             combine_docs_chain_kwargs={"prompt": qa_prompt},
# #             condense_question_prompt=condense_question_prompt,
# #             return_source_documents=True,
# #             verbose=True
# #         )
        
# #         print("System ready for questions!")
        
# #     def chat(self):
# #         """Start an interactive chat session"""
# #         if not self.qa_chain:
# #             print("Initializing system first...")
# #             self.initialize_system()
        
# #         print("\nWelcome to Document Chat! Type 'exit' to end the conversation.")
# #         print("Ask your questions about the document:\n")
        
# #         while True:
# #             question = input("\nYou: ").strip()
            
# #             if question.lower() in ['exit', 'quit', 'bye']:
# #                 print("\nGoodbye!")
# #                 break
                
# #             if not question:
# #                 continue
                
# #             try:
# #                 # Get response from chain
# #                 result = self.qa_chain({
# #                     "question": question
# #                 })
                
# #                 # Extract answer and sources
# #                 answer = result["answer"]
# #                 sources = result.get("source_documents", [])
                
# #                 # Print the response
# #                 print("\nAssistant:", answer)
                
# #                 # Print sources if available
# #                 if sources:
# #                     print("\nSources:")
# #                     for i, source in enumerate(sources, 1):
# #                         page_num = source.metadata.get('page', 'Unknown')
# #                         print(f"\nSource {i} (Page {page_num}):")
# #                         print(source.page_content[:200] + "...")
                
# #             except Exception as e:
# #                 print(f"\nError: Something went wrong. {str(e)}")
# #                 print("Please try asking your question again.")

# # def main():
# #     # Check if file exists
# #     pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
# #     if not os.path.exists(pdf_path):
# #         print(f"Error: File not found at {pdf_path}")
# #         return
        
# #     try:
# #         # Create and start chat system
# #         chat_system = DocumentChat(pdf_path)
# #         chat_system.chat()
        
# #     except Exception as e:
# #         print(f"Error initializing chat system: {str(e)}")

# # if __name__ == "__main__":
# #     main()



# import os
# from src.document_chat import DocumentChat

# def main():
#     # Check if file exists
#     pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
#     if not os.path.exists(pdf_path):
#         print(f"Error: File not found at {pdf_path}")
#         return
        
#     try:
#         # Create and start chat system
#         chat_system = DocumentChat(pdf_path)
#         chat_system.chat()
        
#     except Exception as e:
#         print(f"Error initializing chat system: {str(e)}")

# if __name__ == "__main__":
#     main()




# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Dict, List
# from contextlib import asynccontextmanager
# from src.document_chat import DocumentChat
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Pydantic models
# class QuestionRequest(BaseModel):
#     question: str
#     session_id: str
#     context: Optional[Dict] = None

# class QuestionResponse(BaseModel):
#     answer: str
#     sources: List[Dict[str, str]]
#     session_id: str

# # Initialize FastAPI with startup event
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize the DocumentChat system
#     logger.info("Initializing DocumentChat system...")
#     try:
#         pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
#         app.state.doc_chat = DocumentChat(pdf_path)
#         await app.state.doc_chat.initialize_system()
#         logger.info("DocumentChat system initialized successfully")
#     except Exception as e:
#         logger.error(f"Error initializing DocumentChat: {str(e)}")
#         raise
#     yield
#     # Cleanup (if needed)
#     logger.info("Shutting down...")

# # Create FastAPI app
# app = FastAPI(lifespan=lifespan)

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy"}

# @app.post("/chat", response_model=QuestionResponse)
# async def chat_endpoint(request: QuestionRequest):
#     """Chat endpoint for question answering"""
#     try:
#         logger.info(f"Processing question: {request.question}")
        
#         result = app.state.doc_chat.qa_chain({
#             "question": request.question
#         })
        
#         # Process sources
#         sources = [
#             {
#                 "page": str(source.metadata.get('page', 'Unknown')),
#                 "content": source.page_content[:200] + "..."
#             }
#             for source in result.get("source_documents", [])
#         ]
        
#         response = QuestionResponse(
#             answer=result["answer"],
#             sources=sources,
#             session_id=request.session_id
#         )
        
#         logger.info("Successfully processed question")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Add CORS middleware if needed
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)


# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Dict, List
# from contextlib import asynccontextmanager
# import logging
# from src.document_chat import DocumentChat

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Pydantic models
# class QuestionRequest(BaseModel):
#     question: str
#     session_id: str
#     context: Optional[Dict] = None

# class QuestionResponse(BaseModel):
#     answer: str
#     sources: List[Dict[str, str]]
#     session_id: str

# # Initialize FastAPI with startup event
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize the DocumentChat system
#     logger.info("Initializing DocumentChat system...")
#     try:
#         pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
#         app.state.doc_chat = DocumentChat(pdf_path)
#         await app.state.doc_chat.initialize_system()
#         logger.info("DocumentChat system initialized successfully")
#     except Exception as e:
#         logger.error(f"Error initializing DocumentChat: {str(e)}")
#         raise
#     yield

# app = FastAPI(lifespan=lifespan)

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy"}

# @app.post("/chat", response_model=QuestionResponse)
# async def chat_endpoint(request: QuestionRequest):
#     """Chat endpoint for question answering"""
#     try:
#         logger.info(f"Processing question: {request.question}")
#         result = app.state.doc_chat.qa_chain({
#             "question": request.question
#         })
        
#         # Process sources
#         sources = [
#             {
#                 "page": str(source.metadata.get('page', 'Unknown')),
#                 "content": source.page_content[:200] + "..."
#             }
#             for source in result.get("source_documents", [])
#         ]
        
#         return QuestionResponse(
#             answer=result["answer"],
#             sources=sources,
#             session_id=request.session_id
#         )
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Add CORS middleware if needed
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8080))
#     uvicorn.run(app, host="0.0.0.0", port=port)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import os

# app = FastAPI()

# class HealthCheck(BaseModel):
#     status: str
#     version: str = "1.0.0"

# @app.get("/health", response_model=HealthCheck)
# async def health_check():
#     return HealthCheck(status="healthy")

# @app.get("/")
# async def root():
#     return {"message": "RAG System API is running"}

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 8080))
#     uvicorn.run(app, host="0.0.0.0", port=port)



# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Dict, List
# import logging
# import google.cloud.logging
# from datetime import datetime

# # Setup cloud logging
# client = google.cloud.logging.Client()
# client.setup_logging()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class HealthCheck(BaseModel):
#     status: str
#     version: str = "1.0.0"
#     timestamp: str = datetime.now().isoformat()

# class QuestionRequest(BaseModel):
#     question: str
#     session_id: str
#     context: Optional[Dict] = None

# class QuestionResponse(BaseModel):
#     answer: str
#     sources: List[Dict[str, str]]
#     session_id: str

# app = FastAPI(title="RAG System API")

# @app.get("/health", response_model=HealthCheck)
# async def health_check():
#     logger.info("Health check requested")
#     return HealthCheck(status="healthy")

# @app.get("/")
# async def root():
#     logger.info("Root endpoint accessed")
#     return {"message": "RAG System API is running", "version": "1.0.0"}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8080))
#     uvicorn.run(app, host="0.0.0.0", port=port)


import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import google.cloud.logging
from datetime import datetime
from contextlib import asynccontextmanager
from src.document_chat import DocumentChat

# Setup cloud logging
client = google.cloud.logging.Client()
client.setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthCheck(BaseModel):
    status: str
    version: str = "1.0.0"
    timestamp: str = datetime.now().isoformat()

class QuestionRequest(BaseModel):
    question: str
    session_id: str
    context: Optional[Dict] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    session_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DocumentChat
    logger.info("Initializing RAG system...")
    try:
        pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
        app.state.doc_chat = DocumentChat(pdf_path)
        await app.state.doc_chat.initialize_system()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise
    yield

app = FastAPI(title="RAG System API", lifespan=lifespan)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    logger.info("Health check requested")
    return HealthCheck(status="healthy")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "RAG System API is running", "version": "1.0.0"}

@app.post("/chat", response_model=QuestionResponse)
async def chat(request: QuestionRequest):
    try:
        logger.info(f"Chat request received for session {request.session_id}")
        result = await app.state.doc_chat.get_answer(request.question, request.session_id)
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
