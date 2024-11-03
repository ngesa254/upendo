import os
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

class DocumentChat:
    def __init__(self, pdf_path: str):
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
        
    def initialize_system(self):
        """Initialize the document chat system"""
        print("Loading document...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print("Processing document...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Create chat prompt template
        condense_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant answering questions about a document. 
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Create the chain with proper configuration
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            condense_question_llm=self.llm,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
        )
        
        print("System ready for questions!")
        
    def format_chat_history(self, chat_history):
        """Format chat history into a string"""
        formatted_history = ""
        for message in chat_history:
            if hasattr(message, 'content'):
                if isinstance(message, HumanMessage):
                    formatted_history += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    formatted_history += f"Assistant: {message.content}\n"
        return formatted_history
        
    def chat(self):
        """Start an interactive chat session"""
        if not self.qa_chain:
            print("Initializing system first...")
            self.initialize_system()
        
        print("\nWelcome to Document Chat! Type 'exit' to end the conversation.")
        print("Ask your questions about the document:\n")
        
        while True:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
                
            if not question:
                continue
                
            try:
                # Get response from chain
                result = self.qa_chain.invoke({
                    "question": question
                })
                
                # Extract answer and sources
                answer = result["answer"]
                sources = result.get("source_documents", [])
                
                # Print the response
                print("\nAssistant:", answer)
                
                # Print sources if available
                if sources:
                    print("\nSources:")
                    for i, source in enumerate(sources, 1):
                        page_num = source.metadata.get('page', 'Unknown')
                        print(f"\nSource {i} (Page {page_num}):")
                        print(source.page_content[:200] + "...")
                
            except Exception as e:
                print(f"\nError: Something went wrong. {str(e)}")
                print("Please try asking your question again.")

def main():
    # Check if file exists
    pdf_path = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
        
    try:
        # Create and start chat system
        chat_system = DocumentChat(pdf_path)
        chat_system.chat()
        
    except Exception as e:
        print(f"Error initializing chat system: {str(e)}")

if __name__ == "__main__":
    main()