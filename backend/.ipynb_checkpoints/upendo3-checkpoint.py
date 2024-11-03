import os
import logging
import json
import time
import uuid
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from functools import wraps

# Third-party imports
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Response data structure
@dataclass
class QAResponse:
    answer: str
    confidence_score: float
    sources: List[Dict[str, str]]
    reasoning: str
    relevant_chunks: List[str]

# Custom exceptions
class DocumentProcessingError(Exception):
    """Error during document processing"""
    pass

class ChunkingError(Exception):
    """Error during document chunking"""
    pass

class RetrieverError(Exception):
    """Error during document retrieval"""
    pass

class QAError(Exception):
    """Error during question answering"""
    pass

def setup_logging(log_dir: str = "logs") -> None:
    """Configure logging with both file and console handlers"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_qa_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class ConfidenceScorer:
    """Handles confidence scoring for QA responses"""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        try:
            embedding1 = self.embeddings_model.embed_query(text1)
            embedding2 = self.embeddings_model.embed_query(text2)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logging.warning(f"Error computing semantic similarity: {str(e)}")
            return 0.0
    
    def calculate_confidence(self, 
                           question: str, 
                           answer: str, 
                           context_chunks: List[str],
                           reasoning: str) -> float:
        """Calculate confidence score based on multiple factors"""
        try:
            # Question-context similarity
            context_similarities = [
                self.compute_semantic_similarity(question, chunk) 
                for chunk in context_chunks
            ]
            max_context_similarity = max(context_similarities) if context_similarities else 0
            
            # Answer-context coherence
            answer_context_similarity = max(
                self.compute_semantic_similarity(answer, chunk) 
                for chunk in context_chunks
            ) if context_chunks else 0
            
            # Reasoning quality score
            reasoning_score = min(len(reasoning.split()) / 50.0, 1.0)
            
            # Calculate weighted score
            confidence = (
                0.4 * max_context_similarity +
                0.4 * answer_context_similarity +
                0.2 * reasoning_score
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating confidence score: {str(e)}")
            return 0.0

def retry_api_call(max_attempts: int = 3, initial_wait: float = 1):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=initial_wait, min=4, max=10),
            retry=retry_if_exception_type((
                ConnectionError,
                TimeoutError,
                RetrieverError,
                QAError
            )),
            before_sleep=lambda retry_state: logging.info(
                f"Retrying {func.__name__} - attempt {retry_state.attempt_number}"
            )
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

class DocumentQA:
    """Main class for document question-answering system"""
    
    def __init__(self, pdf_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pdf_path = pdf_path
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize system components with error handling"""
        try:
            self.loader = PyPDFLoader(self.pdf_path)
            self.raw_docs = []
            self.chunks = []
            self.tables = []
            self.vectorstore = None
            self.retriever = None
            self.qa_chain = None
            self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
            self.confidence_scorer = ConfidenceScorer(self.embeddings)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.logger.info("Successfully initialized DocumentQA components")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise DocumentProcessingError(f"Initialization failed: {str(e)}")

    @retry_api_call()
    def load_document(self) -> None:
        """Load document with retry mechanism"""
        try:
            self.logger.info(f"Loading document: {self.pdf_path}")
            self.raw_docs = self.loader.load()
            self.logger.info(f"Successfully loaded {len(self.raw_docs)} pages")
        except Exception as e:
            self.logger.error(f"Document loading failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to load document: {str(e)}")

    def chunk_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Enhanced document chunking with error handling"""
        try:
            self.logger.info("Starting document chunking")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            
            self.chunks = []
            for doc in self.raw_docs:
                try:
                    chunks = text_splitter.split_text(doc.page_content)
                    self.chunks.extend([
                        Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "chunk_size": len(chunk),
                                "source_text": doc.page_content[:200] + "...",
                                "processing_timestamp": datetime.now().isoformat()
                            }
                        )
                        for i, chunk in enumerate(chunks)
                    ])
                except Exception as e:
                    self.logger.warning(f"Error chunking page {doc.metadata.get('page', 'unknown')}: {str(e)}")
                    
            self.logger.info(f"Successfully created {len(self.chunks)} chunks")
            return self.chunks
        except Exception as e:
            self.logger.error(f"Chunking failed: {str(e)}")
            raise ChunkingError(f"Document chunking failed: {str(e)}")

    @retry_api_call()
    def generate_chunk_summaries(self, chunks: List[Document]) -> List[str]:
        """Generate summaries with improved error handling and retries"""
        summaries = []
        batch_size = 5
        
        for i in range(0, len(chunks), batch_size):
            try:
                batch = chunks[i:i + batch_size]
                self.logger.info(f"Processing summary batch {i//batch_size + 1}")
                
                for chunk in batch:
                    try:
                        summary = self._generate_single_summary(chunk.page_content)
                        summaries.append(summary)
                    except Exception as e:
                        self.logger.warning(f"Error summarizing chunk {chunk.metadata.get('chunk_index')}: {str(e)}")
                        summaries.append("Summary generation failed for this chunk")
                        
            except Exception as e:
                self.logger.error(f"Batch processing failed at index {i}: {str(e)}")
                raise
                
        return summaries

    @retry_api_call()
    def _generate_single_summary(self, text: str) -> str:
        """Generate summary for a single chunk with retries"""
        prompt = PromptTemplate.from_template(
            "Summarize the following text chunk for retrieval purposes. Focus on key facts: {text}"
        )
        
        model = VertexAI(
            temperature=0,
            model_name="gemini-pro",
            max_output_tokens=512
        )
        
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"text": text})

    def create_multi_vector_retriever(self, chunk_summaries: List[str]):
        """Create retriever with enhanced chunk handling"""
        try:
            store = InMemoryStore()
            id_key = "doc_id"
            
            self.vectorstore = Chroma(
                collection_name="mm_rag_qa_chunks",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=store,
                id_key=id_key,
            )
            
            # Add documents to retriever
            doc_ids = [str(uuid.uuid4()) for _ in self.chunks]
            summary_docs = [
                Document(
                    page_content=summary,
                    metadata={
                        id_key: doc_ids[i],
                        **self.chunks[i].metadata
                    }
                )
                for i, summary in enumerate(chunk_summaries)
            ]
            
            self.retriever.vectorstore.add_documents(summary_docs)
            self.retriever.docstore.mset(list(zip(doc_ids, self.chunks)))
            
        except Exception as e:
            self.logger.error(f"Failed to create retriever: {str(e)}")
            raise RetrieverError(f"Failed to create retriever: {str(e)}")

    def setup_qa_chain(self):
        """Set up enhanced QA chain with reasoning"""
        try:
            qa_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", """You are an AI assistant helping to answer questions about a document.
                Use the following context to answer the question. If you don't know the answer,
                say that you don't know. Don't try to make up an answer.
                
                Context: {context}"""),
                ("human", "{question}"),
                ("system", """Please provide your response in the following JSON format:
                {
                    "answer": "Your detailed answer here",
                    "reasoning": "Step by step explanation of how you arrived at the answer",
                    "relevant_sections": ["List of specific sections from context that support your answer"]
                }""")
            ])
            
            llm = ChatVertexAI(
                model_name="gemini-pro",
                temperature=0.7,
                max_output_tokens=1024
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
                return_generated_question=True,
                output_key="answer"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup QA chain: {str(e)}")
            raise QAError(f"Failed to setup QA chain: {str(e)}")

    def initialize_system(self):
        """Initialize the complete QA system"""
        try:
            self.logger.info("Starting system initialization")
            
            # Load document
            self.load_document()
            
            # Chunk documents
            self.chunk_documents()
            
            # Generate summaries
            chunk_summaries = self.generate_chunk_summaries(self.chunks)
            
            # Create retriever
            self.create_multi_vector_retriever(chunk_summaries)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            self.logger.info("System initialization complete")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            raise

    @retry_api_call()
    def ask_question(self, question: str) -> QAResponse:
        """Process question and return response with confidence scoring"""
        if not self.qa_chain:
            raise ValueError("QA system not initialized. Call initialize_system() first.")
            
        self.logger.info(f"Processing question: {question}")
        
        try:
            # Get response
            response = self.qa_chain.invoke({
                "question": question
            })
            
            self.logger.info("Successfully received response from QA chain")
            
            # Parse response
            try:
                parsed_response = self._parse_qa_response(response)
            except Exception as e:
                self.logger.warning(f"Error parsing response: {str(e)}")
                parsed_response = self._create_fallback_response(response)
            
            # Create final response
            qa_response = self._create_qa_response(
                question, parsed_response, response["source_documents"]
            )
            
            self.logger.info(f"Question processed successfully. Confidence: {qa_response.confidence_score:.2f}")
            return qa_response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}\n{traceback.format_exc()}")
            raise QA