import os
from typing import List
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

class DocumentQA:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()
        self.texts = [d.page_content for d in self.docs]
        self.tables = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def generate_text_summaries(self, summarize_texts=False):
        """Generate summaries of text elements"""
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. 
        Table or text: {element}"""
        
        prompt = PromptTemplate.from_template(prompt_text)
        empty_response = RunnableLambda(lambda x: AIMessage(content="Error processing document"))
        
        model = VertexAI(
            temperature=0,
            model_name="gemini-pro",
            max_output_tokens=1024
        ).with_fallbacks([empty_response])
        
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        text_summaries = []
        table_summaries = []
        
        if self.texts and summarize_texts:
            text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 1})
        elif self.texts:
            text_summaries = self.texts
            
        if self.tables:
            table_summaries = summarize_chain.batch(self.tables, {"max_concurrency": 1})
            
        return text_summaries, table_summaries
    
    def create_multi_vector_retriever(self, text_summaries: List[str], table_summaries: List[str]):
        """Create retriever that indexes summaries"""
        store = InMemoryStore()
        id_key = "doc_id"
        
        # Initialize vectorstore with embeddings
        self.vectorstore = Chroma(
            collection_name="mm_rag_qa",
            embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@latest")
        )
        
        # Create retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=id_key,
        )
        
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
            
        if text_summaries:
            add_documents(self.retriever, text_summaries, self.texts)
        if table_summaries:
            add_documents(self.retriever, table_summaries, self.tables)
            
    def setup_qa_chain(self):
        """Set up the question-answering chain with conversation memory"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom prompt template for better context utilization
        qa_prompt = PromptTemplate(
            template="""You are an AI assistant helping to answer questions about a document.
            Use the following context to answer the question. If you don't know the answer,
            say that you don't know. Don't try to make up an answer.
            
            Context: {context}
            
            Chat History: {chat_history}
            Question: {question}
            
            Please provide a detailed answer with specific information from the context when available:""",
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
        """Initialize the complete QA system"""
        # Generate summaries
        text_summaries, table_summaries = self.generate_text_summaries(summarize_texts=True)
        
        # Create retriever
        self.create_multi_vector_retriever(text_summaries, table_summaries)
        
        # Setup QA chain
        self.setup_qa_chain()
        
    def ask_question(self, question: str) -> dict:
        """Ask a question and get an answer with sources"""
        if not self.qa_chain:
            raise ValueError("QA system not initialized. Call initialize_system() first.")
            
        response = self.qa_chain({"question": question})
        
        # Format the response
        answer = {
            "answer": response["answer"],
            "sources": [doc.page_content[:200] + "..." for doc in response["source_documents"]]
        }
        
        return answer

# Example usage
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
    
    # Ask questions
    for question in questions:
        print(f"\nQuestion: {question}")
        response = qa_system.ask_question(question)
        print("\nAnswer:", response["answer"])
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"\nSource {i}:", source)

if __name__ == "__main__":
    main()