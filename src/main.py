from fastapi import FastAPI

from src.agent.graph import graph

app = FastAPI()


@app.get("/invoke")
def invoke():
    print(graph)
    return {"message": "Graph invoked"}
