import os
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

load_dotenv(find_dotenv())

# create the index
index_name = "umag-hybrid-search"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if index_name in pc.list_indexes().names():
    pc.delete_index(name=index_name)

pc.create_index(
    name=index_name,
    dimension=3072,  # dimensionality of dense model
    metric="dotproduct",  # sparse values supported only for dotproduct
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(index_name)

bm25_encoder = BM25Encoder.default()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
)

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k=10
)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path

docs_folder = Path(
    r"D:\HKU\Inno Wing RA\UBC Exchange\self_rag\output\Objectifying_China\docs"
)
loader = DirectoryLoader(
    str(docs_folder.absolute()),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()

# assign id to docs
from hashlib import sha256


def hash_content(content: str):
    hash_obj = sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


doc_ids = [hash_content(doc.page_content) for doc in docs]
metadatas = [{"doc_id": id} for id in doc_ids]
retriever.add_texts(
    texts=[d.page_content for d in docs], ids=doc_ids, metadatas=metadatas
)
