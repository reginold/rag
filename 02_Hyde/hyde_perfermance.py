from datasets import load_dataset
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
import os
import sys

# Load the env
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (applied_rag) by navigating up one level from the current script
parent_dir = os.path.dirname(current_script_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import settings


######################### Method 1
# to load all train, dev and test sets
dataset = load_dataset('castorini/mr-tydi', "japanese", split="train")
tydi_df = pd.DataFrame(dataset).sample(100, random_state=42)
for col in ["positive_passages", "negative_passages"]:
    tydi_df[col] = tydi_df[col].apply(lambda x: x[0]["text"])
tydi_df_sample = tydi_df.iloc[:50,:].copy()

# print(tydi_df_sample.head(2))

from langchain.vectorstores import FAISS
from tqdm.auto import tqdm

def get_rank(query, docs):
    for i, doc in enumerate(docs, start=1):
        if query == doc.metadata["query"]:
            return i

def test(test_query_list, vectorstore):
    # fetch the documents
    rank_list = []
    for title in tqdm(test_query_list):
        docs = vectorstore.similarity_search(title, k=200)
        rank_list.append(get_rank(title, docs))

    # summarize the results
    return rank_list

def get_mrr(rank_list):
    return sum([1/rank for rank in rank_list])/len(rank_list)
def get_correct_num(rank_list):
    return len([rank for rank in rank_list if rank == 1])

# prepare the vectorstore
docs = tydi_df["positive_passages"].tolist() + tydi_df["negative_passages"].tolist()
meta_datas = [{"query": q} for q in tydi_df["query"].tolist()] + [{"query": ""} for q in tydi_df["query"].tolist()]
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(
    texts=docs,
    embedding=base_embeddings,
    metadatas=meta_datas,
)

rank_list = test(tydi_df_sample["query"].tolist(), vectorstore)

print(f"mrr: {get_mrr(rank_list):.3f}")
print(f"correct num: {get_correct_num(rank_list)}")

######################### Method 2
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

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

prompt_template = """質問に答えてください。
質問：{question}
答案："""

llm = AzureChatOpenAI(
    deployment_name=config.deployment_name,
    openai_api_key=config.openai_api_key,
    openai_api_base=config.openai_api_base,
    openai_api_type=config.openai_api_type,
    openai_api_version=config.openai_api_version,
    model_name=config.model_name,
    temperature=0.0
)
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=base_embeddings)
vectorstore.embedding_function = embeddings.embed_query

hyde_rank_list = test(tydi_df_sample["query"].tolist(), vectorstore)
print(f"mrr: {get_mrr(hyde_rank_list):.3f}")
print(f"correct num: {get_correct_num(hyde_rank_list)}")


######################### Method 3
class HyDEWithTitle(HypotheticalDocumentEmbedder):

    def embed_query(self, text: str):
        """Generate a hypothetical document and embedded it."""
        var_name = self.llm_chain.input_keys[0]
        result = self.llm_chain.generate([{var_name: text}])
        documents = [generation.text for generation in result.generations[0]]
        # add query to the beginning of the document
        documents = [f"{text}\n{document}" for document in documents]
        embeddings = self.embed_documents(documents)
        return self.combine_embeddings(embeddings)

embeddings = HyDEWithTitle(llm_chain=llm_chain, base_embeddings=base_embeddings)
vectorstore.embedding_function = embeddings.embed_query

hyde_with_title_rank_list = test(tydi_df_sample["query"].tolist(), vectorstore)

print(f"mrr: {get_mrr(hyde_with_title_rank_list):.3f}")
print(f"correct num: {get_correct_num(hyde_with_title_rank_list)}")


################ Result
# mrr: 0.187
# correct num: 7
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:51<00:00,  1.02s/it]
# mrr: 0.243
# correct num: 10
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:52<00:00,  1.06s/it]
# mrr: 0.284
# correct num: 11