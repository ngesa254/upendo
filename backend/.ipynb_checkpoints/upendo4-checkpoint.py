import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
            return_messages=True
        )
        self.qa_chain = None
        
    def initialize_system(self):
        """Initialize the document chat system"""
        print("Loading document...")
        # Load the document
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print("Processing document...")
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        print("Creating vector store...")
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Create QA chain
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", """You are a helpful AI assistant answering questions about a document.
            Use the following context to answer the question. If you don't know the answer,
            say that you don't know. Base your answer only on the provided context.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        print("System ready for questions!")
        
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
                # Get response from QA chain
                response = self.qa_chain.invoke({"question": question})
                print("\nAssistant:", response["answer"])
                
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