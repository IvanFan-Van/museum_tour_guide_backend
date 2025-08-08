import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

load_dotenv(find_dotenv())

### Query Router
llm = AzureChatOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYEMENT"],
)

zero_temp_llm = llm = AzureChatOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYEMENT"],
    temperature=0.0,
)


def format_documents_as_string(docs: list[Document]) -> str:
    if len(docs) == 0:
        return ""

    doc_string_list = []
    for doc in docs:
        metadata_list = []
        doc_id = "doc_id: " + doc.metadata.get("doc_id", "")
        metadata_list.append(doc_id)
        if "relevance_score" in doc.metadata:
            score = "score: " + str(doc.metadata.get("relevance_score", ""))
            metadata_list.append(score)
        metadata_header = "[" + " | ".join(metadata_list) + "]"
        doc_string = metadata_header + "\n" + doc.page_content
        doc_string_list.append(doc_string)

    return "\n\n".join(doc_string_list)
