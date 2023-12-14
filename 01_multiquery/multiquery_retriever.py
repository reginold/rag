import os
import logging
from typing import List
from pydantic import BaseModel, Field

from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import sys

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (applied_rag) by navigating up one level from the current script
parent_dir = os.path.dirname(current_script_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import settings
from openai import AzureOpenAI

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

# Load and split blog post
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# Initialize VectorDB with embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

# Pydantic model for line list
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

# Output parser for LLMChain
class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()

# Prompt template for queries
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}"""  # Template content goes here
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

# Setup logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Initialize LLMChain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Initialize MultiQueryRetriever
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)

# Retrieve documents
unique_docs = retriever.get_relevant_documents(
    query="What does the course say about regression?"
)
print(len(unique_docs))
