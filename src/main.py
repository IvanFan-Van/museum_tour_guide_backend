from typing import TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from src.agent.graph import graph

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(TypedDict):
    query: str


@app.post("/invoke")
def invoke(req: QueryRequest):
    messages = [HumanMessage(content=req["query"])]
    response = graph.invoke({"messages": messages})
    # print(response)
    return response.get("generation", None)
