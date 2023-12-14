# retrivel from PDF
import os
import sys
from pydantic import BaseModel, Field
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from openai import AzureOpenAI

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (applied_rag) by navigating up one level from the current script
parent_dir = os.path.dirname(current_script_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import settings
from langchain.vectorstores import LanceDB
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Configuration data class
class AppConfig(BaseModel):
    openai_api_key: str
    openai_api_base: str
    openai_api_type: str
    deployment_name: str
    model_name: str
    openai_api_version: str

# Load configuration from settings
config = AppConfig(
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_api_base,
    openai_api_type=settings.openai_api_type,
    deployment_name=settings.deployment_name,
    model_name=settings.model_name,
    openai_api_version=settings.openai_api_version
)

# Initialize Azure LLM
llm = AzureChatOpenAI(
    deployment_name=config.deployment_name,
    openai_api_key=config.openai_api_key,
    openai_api_base=config.openai_api_base,
    openai_api_type=config.openai_api_type,
    openai_api_version=config.openai_api_version,
    model_name=config.model_name,
    temperature=0.0
)

emebeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = HypotheticalDocumentEmbedder.from_llm(llm, emebeddings, "web_search")


prompt_template = """
As a knowledgeable and helpful research assistant, your task is to provide informative answers based on the given context. Use your extensive knowledge base to offer clear, concise, and accurate responses to the user's inquiries.
if quetion is not related to documents simply say you dont know
Question: {question}

Answer:
"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embeddings
)

# Loading data from pdf
pdf_folder_path = '/workspaces/rag/02_Hyde/pamphlet.pdf'

loader = PyPDFLoader(pdf_folder_path)
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=50,
)
documents = text_splitter.split_documents(docs)



# lancedb as vectorstore
db = lancedb.connect('/tmp/lancedb')
table = db.create_table("documentsai", data=[
    {"vector": embeddings.embed_query("アジャイル"), "text": "アジャイル", "id": "1"}
], mode="overwrite")
vector_store = LanceDB.from_documents(documents, embeddings, connection=table)

# passing in the string query to get some refrence
# query = "which factors appear to be the major nutritional limitations of fast-food meals"
query = "Ridgelinezアジャイルサービスとその優位性を説明してください。"

# result = vector_store.similarity_search(query)
# print(result)

answer = llm_chain.run(query)
print(answer)