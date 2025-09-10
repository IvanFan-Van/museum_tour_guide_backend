from typing import Annotated, List, Literal

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    get_buffer_string,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.pinecone_retriever import retriever
from agent.prompts import GENERATOR_PROMPT
from agent.query_router import QueryRouting, query_router
from agent.question_rewriter import question_rewriter
from agent.retrieval_grader import GradeDocuments, retrieval_grader
from agent.utils import format_documents_as_string, llm
from agent.web_search import wiki_search

### Generate
# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_PROMPT),
        (
            "human",
            "## Chat History\n{chat_history}\n\n"
            "## Retrieved Context\n{documents}\n\n"
            "## Current User Query\n{user_query}\n\n"
            "## Response\n",
        ),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()


### Question Re-writer
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: The history of messages.
        user_query: The original user query.
        current_query: The current query being processed (can be rewritten).
        db_documents: Documents retrieved from the database.
        web_documents: Documents retrieved from the web search.
        final_documents: The final list of documents used for generation.
        generation: The LLM's generated response.
        retries_left: The number of retries left for query transformation.
        keywords: Keywords extracted for web search.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    current_query: str
    db_documents: List[Document]
    web_documents: List[Document]
    final_documents: List[Document]
    generation: str
    retries_left: int
    keywords: List[str]
    grade_documents: Literal["db_documents", "web_documents"]


### Nodes
def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    query = state["current_query"]

    # Retrieval
    documents = retriever.invoke(query)
    return {"db_documents": documents}


def fuse_documents(state: GraphState):
    """
    Fuse Documents from grade documents
    """
    db_documents = state.get("db_documents", [])
    web_documents = state.get("web_documents", [])
    documents = db_documents + web_documents

    filtered_docs = []
    for doc in documents:
        assert "relevance_score" in doc.metadata
        if doc.metadata["relevance_score"] >= 0.7:
            filtered_docs.append(doc)

    return {"final_documents": filtered_docs}


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    db_documents = state["db_documents"]
    web_documents = state["web_documents"]
    documents = db_documents + web_documents
    messages = state["messages"]
    question = state["current_query"]
    chat_history = get_buffer_string(messages)
    documents_string = format_documents_as_string(documents)

    # RAG generation
    generation = rag_chain.invoke(
        {
            "documents": documents_string,
            "user_query": question,
            "chat_history": chat_history,
        }
    )
    return {
        "messages": [AIMessage(content=generation)],
        "generation": generation,
    }


def direct_generate(state: GraphState):
    """
    Generate answer without documents
    """
    messages = state["messages"]
    query = state["current_query"]

    chat_history = get_buffer_string(messages)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_PROMPT),
            (
                "human",
                "Here is the chat history:\n\n {chat_history} \n\n"
                "Here is the user query:\n\n {query}",
            ),
        ]
    )

    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"query": query, "chat_history": chat_history})
    return {"messages": [AIMessage(content=answer)], "generation": answer}


def grade_documents(state: GraphState):
    """
    Score documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["current_query"]
    grade_documents = state["grade_documents"]
    assert grade_documents in state
    documents: list[Document] = state[grade_documents]

    # Score each doc
    formatted_blocks = format_documents_as_string(documents)
    user_prompt = (
        f'Here is the query: "{question}"\n\n'
        "Here are the retrieved text blocks:\n"
        f"{formatted_blocks}\n\n"
        f"You should provide exactly {len(documents)} rankings, in order."
    )

    def get_scores(user_prompt: str):
        scores = GradeDocuments.model_validate(
            retrieval_grader.invoke(
                {
                    "user_prompt": user_prompt,
                }
            )
        )
        return scores

    scores = get_scores(user_prompt)
    for idx, score in enumerate(scores.block_rankings):
        documents[idx].metadata["relevance_score"] = score.relevance_score
        documents[idx].metadata["reasoning"] = score.reasoning

    return {grade_documents: documents}


def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    original_question = state["user_query"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": original_question})
    return {"current_query": better_question, "retries_left": state["retries_left"] - 1}


def web_search(state: GraphState):
    query = state["current_query"]
    if not isinstance(query, str):
        raise ValueError("message content must be a string")
    web_results = wiki_search(query)
    web_docs = web_results["documents"]
    keywords = web_results["keywords"]
    return {
        "web_documents": web_docs,
        "keywords": keywords,
        "grade_documents": "web_documents",
    }


def init(state: GraphState):
    """Initialize the state for a new run"""
    print("---INITIALIZING RUN---")
    question = state["messages"][-1].content
    assert isinstance(question, str)
    # This node returns a dictionary to update the state
    return {
        "user_query": question,
        "current_query": question,
        "retries_left": 2,
        "db_documents": [],
        "web_documents": [],
        "final_documents": [],
        "keywords": [],
        "grade_documents": "db_documents",
    }


### Edges
def decide_to_retrieve(state: GraphState):
    """
    Determine whether to generate an answer, or going through the RAG pipeline

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    question = state["current_query"]

    routing = QueryRouting.model_validate(query_router.invoke({"question": question}))
    if routing.need_rag:
        return "need_rag"
    else:
        return "no_need_rag"


def decide_to_generate(state: GraphState):
    """
    Determine whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["final_documents"]
    retries_left = state["retries_left"]
    if len(filtered_documents) < 3 and retries_left > 0:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    elif (
        len(filtered_documents) < 3  # very limited context
        and retries_left == 0  # no retries
        and state.get("grade_documents") != "web_documents"  # no web search before
    ):
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("init", init)  # init
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("fuse_documents", fuse_documents)  # fuse documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("direct_generate", direct_generate)  # direct_generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)

# Build graph
workflow.add_edge(START, "init")
workflow.add_conditional_edges(
    "init",
    decide_to_retrieve,
    {"need_rag": "retrieve", "no_need_rag": "direct_generate"},
)
workflow.add_edge("direct_generate", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "fuse_documents")
workflow.add_conditional_edges(
    "fuse_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "grade_documents")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
