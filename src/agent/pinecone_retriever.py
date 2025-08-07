import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

load_dotenv(find_dotenv())

# create the index
index_name = "umag-hybrid-search"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
if index_name not in pc.list_indexes().names():
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
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k=5
)
