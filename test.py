import os
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI

load_dotenv(find_dotenv())

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = "gpt-4o"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)
