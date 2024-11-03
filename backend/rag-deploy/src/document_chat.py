# import os
# from typing import Dict, Any
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.chroma import Chroma
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory

# class DocumentChat:
#     def __init__(self, pdf_path: str):
#         self.pdf_path = pdf_path
#         self.llm = ChatVertexAI(
#             model_name="gemini-pro",
#             temperature=0.7,
#             max_output_tokens=1024
#         )
#         self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             output_key="answer"
#         )
#         self.qa_chain = None
        
#     def initialize_system(self):
#         """Initialize the document chat system"""
#         print("Loading document...")
#         loader = PyPDFLoader(self.pdf_path)
#         documents = loader.load()
        
#         print("Processing document...")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100
#         )
#         splits = text_splitter.split_documents(documents)
        
#         print("Creating vector store...")
#         vectorstore = Chroma.from_documents(
#             documents=splits,
#             embedding=self.embeddings
#         )
        
#         # Create the condense question prompt
#         condense_question_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:""")
        
#         # Create the QA prompt with explicit input variables
#         qa_prompt = PromptTemplate(
#             template="""Use the following pieces of context to answer the question at the end.
            
# Context:
# {context}

# Question: {question}

# Answer:""",
#             input_variables=["context", "question"]
#         )
        
#         # Create the chain with proper configuration
#         self.qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#             memory=self.memory,
#             get_chat_history=lambda h: str(h),
#             combine_docs_chain_kwargs={"prompt": qa_prompt},
#             condense_question_prompt=condense_question_prompt,
#             return_source_documents=True,
#             verbose=True
#         )
        
#         print("System ready for questions!")
        
#     def chat(self):
#         """Start an interactive chat session"""
#         if not self.qa_chain:
#             print("Initializing system first...")
#             self.initialize_system()
        
#         print("\nWelcome to Document Chat! Type 'exit' to end the conversation.")
#         print("Ask your questions about the document:\n")
        
#         while True:
#             question = input("\nYou: ").strip()
            
#             if question.lower() in ['exit', 'quit', 'bye']:
#                 print("\nGoodbye!")
#                 break
                
#             if not question:
#                 continue
                
#             try:
#                 # Get response from chain
#                 result = self.qa_chain({
#                     "question": question
#                 })
                
#                 # Extract answer and sources
#                 answer = result["answer"]
#                 sources = result.get("source_documents", [])
                
#                 # Print the response
#                 print("\nAssistant:", answer)
                
#                 # Print sources if available
#                 if sources:
#                     print("\nSources:")
#                     for i, source in enumerate(sources, 1):
#                         page_num = source.metadata.get('page', 'Unknown')
#                         print(f"\nSource {i} (Page {page_num}):")
#                         print(source.page_content[:200] + "...")
                
#             except Exception as e:
#                 print(f"\nError: Something went wrong. {str(e)}")
#                 print("Please try asking your question again.")




import os
from typing import Dict, Any, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

logger = logging.getLogger(__name__)

class DocumentChat:
    def __init__(self, pdf_path: str):
        logger.info(f"Initializing DocumentChat with PDF: {pdf_path}")
        self.pdf_path = pdf_path
        self.llm = ChatVertexAI(
            model_name="gemini-pro",
            temperature=0.7,
            max_output_tokens=1024
        )
        self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
    async def initialize_system(self):
        """Initialize the document chat system"""
        try:
            logger.info("Loading document...")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            logger.info("Processing document...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            logger.info("Creating vector store...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            logger.info("Setting up QA chain...")
            qa_prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know.

                Context: {context}

                Question: {question}

                Answer:""",
                input_variables=["context", "question"]
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
            )
            
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise
            
    async def get_answer(self, question: str, session_id: str) -> Dict[str, Any]:
        """Get answer for a question"""
        try:
            if not self.qa_chain:
                logger.error("QA chain not initialized")
                raise ValueError("System not initialized")
                
            logger.info(f"Processing question for session {session_id}: {question}")
            
            result = self.qa_chain({"question": question})
            
            sources = [
                {
                    "page": str(doc.metadata.get("page", "Unknown")),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in result.get("source_documents", [])
            ]
            
            return {
                "answer": result["answer"],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise
