import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

### Query Router
llm = AzureChatOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYEMENT"],
)
