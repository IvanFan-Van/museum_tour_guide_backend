from typing import TypedDict

from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from src.agent.graph import graph

app = FastAPI()


class QueryRequest(TypedDict):
    query: str


@app.post("/invoke")
def invoke(req: QueryRequest):
    messages = [HumanMessage(content=req["query"])]
    response = graph.invoke({"messages": messages})
    # print(response)
    return response.get("generation", None)
