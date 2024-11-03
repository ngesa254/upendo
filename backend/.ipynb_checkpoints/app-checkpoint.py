import os

# Define the file path
directory = "data"
file_name = "Africa_Developer_Ecosystem_Report_2021.pdf"
source = os.path.join(directory, file_name)

# Print the absolute path for verification
absolute_path = os.path.abspath(source)
print("Looking for file at:", absolute_path)

# Check if the file exists at the absolute path
if os.path.isfile(absolute_path):
    print("File found:", absolute_path)
    
    
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(absolute_path)
docs = loader.load()
tables = []
texts = [d.page_content for d in docs]

from langchain_google_vertexai import VertexAI , ChatVertexAI , VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=False):
   """
   Summarize text elements
   texts: List of str
   tables: List of str
   summarize_texts: Bool to summarize texts
   """

   # Prompt
   prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
   These summaries will be embedded and used to retrieve the raw text or table elements. \
   Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
   prompt = PromptTemplate.from_template(prompt_text)
   empty_response = RunnableLambda(
       lambda x: AIMessage(content="Error processing document")
   )
   # Text summary chain
   model = VertexAI(
       temperature=0, model_name="gemini-pro", max_output_tokens=1024
   ).with_fallbacks([empty_response])
   summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

   # Initialize empty summaries
   text_summaries = []
   table_summaries = []

   # Apply to text if texts are provided and summarization is requested
   if texts and summarize_texts:
       text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
   elif texts:
       text_summaries = texts

   # Apply to tables if tables are provided
   if tables:
       table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

   return text_summaries, table_summaries


# Get text summaries
text_summaries, table_summaries = generate_text_summaries(
   texts, tables, summarize_texts=True
)

text_summaries[0]

import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def create_multi_vector_retriever(
   vectorstore, text_summaries, texts, table_summaries, tables
):
   """
   Create retriever that indexes summaries
   """

   # Initialize the storage layer
   store = InMemoryStore()
   id_key = "doc_id"

   # Create the multi-vector retriever
   retriever = MultiVectorRetriever(
       vectorstore=vectorstore,
       docstore=store,
       id_key=id_key,
   )

   # Helper function to add documents to the vectorstore and docstore
   def add_documents(retriever, doc_summaries, doc_contents):
       doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
       summary_docs = [
           Document(page_content=s, metadata={id_key: doc_ids[i]})
           for i, s in enumerate(doc_summaries)
       ]
       retriever.vectorstore.add_documents(summary_docs)
       retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

   # Add texts, tables, and images
   # Check that text_summaries is not empty before adding
   if text_summaries:
       add_documents(retriever, text_summaries, texts)
   # Check that table_summaries is not empty before adding
   if table_summaries:
       add_documents(retriever, table_summaries, tables)

   return retriever


# The vectorstore to use to index the summaries
vectorstore = Chroma(
   collection_name="mm_rag_cj_blog",
   embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko@latest"),
)

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
   vectorstore,
   text_summaries,
   texts,
   table_summaries,
   tables
)
 