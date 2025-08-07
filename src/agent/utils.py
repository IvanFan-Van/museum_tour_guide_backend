import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv(find_dotenv())

### Query Router
llm = AzureChatOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYEMENT"],
)
