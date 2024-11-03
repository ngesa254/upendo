import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class QAResponse:
    answer: str
    confidence_score: float
    sources: List[Dict[str, str]]
    reasoning: str
    relevant_chunks: List[str]

class ConfidenceScorer:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        
    def compute_semantic_similarity(self, question: str, context: str) -> float:
        """Compute semantic similarity between question and context"""
        question_embedding = self.embeddings_model.embed_query(question)
        context_embedding = self.embeddings_model.embed_query(context)
        similarity = cosine_similarity(
            [question_embedding], 
            [context_embedding]
        )[0][0]
        return float(similarity)
    
    def calculate_confidence(self, 
                           question: str, 
                           answer: str, 
                           context_chunks: List[str],
                           reasoning: str) -> float:
        """Calculate confidence score based on multiple factors"""
        # Semantic similarity between question and best context chunk
        context_similarities = [
            self.compute_semantic_similarity(question, chunk) 
            for chunk in context_chunks
        ]
        max_context_similarity = max(context_similarities) if context_similarities else 0
        
        # Answer consistency with context
        answer_context_similarity = max(
            self.compute_semantic_similarity(answer, chunk) 
            for chunk in context_chunks
        ) if context_chunks else 0
        
        # Reasoning quality (length and structure)
        reasoning_score = min(len(reasoning.split()) / 50.0, 1.0)
        
        # Weighted combination of factors
        confidence = (
            0.4 * max_context_similarity +
            0.4 * answer_context_similarity +
            0.2 * reasoning_score
        )
        
        return min(confidence, 1.0)

class DocumentQA:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.raw_docs = self.loader.load()
        self.chunks = []
        self.tables = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
        self.confidence_scorer = ConfidenceScorer(self.embeddings)
        
    def chunk_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split documents into chunks for better retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        # Split documents into chunks
        self.chunks = []
        for doc in self.raw_docs:
            chunks = text_splitter.split_text(doc.page_content)
            # Preserve page numbers and other metadata
            self.chunks.extend([
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "source_text": doc.page_content[:200] + "..."
                    }
                )
                for i, chunk in enumerate(chunks)
            ])
        
        return self.chunks
    
    def generate_chunk_summaries(self, chunks: List[Document]) -> List[str]:
        """Generate summaries for document chunks"""
        prompt_text = """Summarize the following text chunk for retrieval purposes. 
        Focus on key facts and main points: {chunk}"""
        
        prompt = PromptTemplate.from_template(prompt_text)
        empty_response = RunnableLambda(
            lambda x: AIMessage(content="Error processing chunk")
        )
        
        model = VertexAI(
            temperature=0,
            model_name="gemini-pro",
            max_output_tokens=512
        ).with_fallbacks([empty_response])
        
        summarize_chain = {"chunk": lambda x: x} | prompt | model | StrOutputParser()
        
        # Process chunks in batches
        summaries = []
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_summaries = summarize_chain.batch(
                [doc.page_content for doc in batch],
                {"max_concurrency": 1}
            )
            summaries.extend(batch_summaries)
            
        return summaries

    def create_multi_vector_retriever(self, chunk_summaries: List[str]):
        """Create retriever with enhanced chunk handling"""
        store = InMemoryStore()
        id_key = "doc_id"
        
        self.vectorstore = Chroma(
            collection_name="mm_rag_qa_chunks",
            embedding_function=self.embeddings
        )
        
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=id_key,
        )
        
        # Add chunked documents
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
    
    def setup_qa_chain(self):
        """Set up enhanced QA chain with reasoning"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_prompt = PromptTemplate(
            template="""You are an AI assistant helping to answer questions about a document.
            Use the following context to answer the question. If you don't know the answer,
            say that you don't know. Don't try to make up an answer.
            
            Context: {context}
            
            Chat History: {chat_history}
            Question: {question}
            
            Please provide your response in the following JSON format:
            {{
                "answer": "Your detailed answer here",
                "reasoning": "Step by step explanation of how you arrived at the answer",
                "relevant_sections": ["List of specific sections from context that support your answer"]
            }}
            """,
            input_variables=["context", "chat_history", "question"]
        )
        
        llm = ChatVertexAI(
            model_name="gemini-pro",
            temperature=0.7,
            max_output_tokens=1024
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def initialize_system(self):
        """Initialize the enhanced QA system"""
        # Chunk documents
        print("Chunking documents...")
        self.chunk_documents()
        
        # Generate summaries for chunks
        print("Generating chunk summaries...")
        chunk_summaries = self.generate_chunk_summaries(self.chunks)
        
        # Create retriever
        print("Creating retriever...")
        self.create_multi_vector_retriever(chunk_summaries)
        
        # Setup QA chain
        print("Setting up QA chain...")
        self.setup_qa_chain()
        
        print("System initialization complete!")
    
    def ask_question(self, question: str) -> QAResponse:
        """Ask a question and get an enhanced response with confidence scoring"""
        if not self.qa_chain:
            raise ValueError("QA system not initialized. Call initialize_system() first.")
            
        # Get response from QA chain
        response = self.qa_chain({"question": question})
        
        try:
            # Parse the JSON response
            parsed_response = json.loads(response["answer"])
            answer = parsed_response["answer"]
            reasoning = parsed_response["reasoning"]
            relevant_sections = parsed_response["relevant_sections"]
        except json.JSONDecodeError:
            # Fallback if response isn't in expected JSON format
            answer = response["answer"]
            reasoning = "Direct response provided"
            relevant_sections = []
        
        # Get relevant chunks from source documents
        relevant_chunks = [
            doc.page_content
            for doc in response["source_documents"]
        ]
        
        # Calculate confidence score
        confidence_score = self.confidence_scorer.calculate_confidence(
            question=question,
            answer=answer,
            context_chunks=relevant_chunks,
            reasoning=reasoning
        )
        
        # Prepare sources with metadata
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc in response["source_documents"]
        ]
        
        return QAResponse(
            answer=answer,
            confidence_score=confidence_score,
            sources=sources,
            reasoning=reasoning,
            relevant_chunks=relevant_sections
        )

def main():
    # Initialize the system
    pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
    qa_system = DocumentQA(pdf_path)
    qa_system.initialize_system()
    
    # Example questions
    questions = [
        "What are the key findings about developer demographics in Africa?",
        "What are the main challenges faced by African developers?",
        "What are the most popular programming languages among African developers?"
    ]
    
    # Ask questions and show enhanced output
    for question in questions:
        print(f"\nQuestion: {question}")
        response = qa_system.ask_question(question)
        print("\nAnswer:", response.answer)
        print(f"\nConfidence Score: {response.confidence_score:.2f}")
        print("\nReasoning:", response.reasoning)
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"\nSource {i}:")
            print(f"Page: {source['page']}")
            print(f"Chunk Index: {source['chunk_index']}")
            print(f"Content: {source['content']}")
        print("\nRelevant Sections:")
        for section in response.relevant_chunks:
            print(f"- {section}")

if __name__ == "__main__":
    main()