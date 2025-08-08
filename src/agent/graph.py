from pprint import pprint
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.answer_grader import GradeAnswer, answer_grader
from agent.hallucination_grader import GradeHallucinations, hallucination_grader
from agent.pinecone_retriever import retriever
from agent.prompts import GENERATOR_PROMPT
from agent.query_router import QueryRouting, query_router
from agent.question_rewriter import question_rewriter
from agent.retrieval_grader import GradeDocuments, retrieval_grader
from agent.utils import format_documents_as_string, llm

### Generate
# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_PROMPT),
        (
            "human",
            "**Now, respond to this:**\n"
            "**User Query:** '{user_query}'\n"
            "**Relevant Context:** {documents}\n"
            "**Chat History: {chat_history}**\n",
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
        question: question
        generation: LLM generation
        documents: list of documents
    """

    documents: List[str]
    messages: Annotated[list[AnyMessage], add_messages]


### Nodes
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    query = state["messages"][-1].content

    # Retrieval
    documents = retriever.invoke(query)
    return {"documents": documents}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    documents = state["documents"]
    messages = state["messages"]
    question = messages[-1].content
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
    }


def direct_generate(state):
    """
    Generate answer without documents
    """
    messages = state["messages"]
    query = messages[-1].content

    chat_history = "\n\n".join([m.content for m in messages])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_PROMPT),
            (
                "human",
                "Here is the chat history: {chat_history}\n\n"
                "Here is the user query: {query}",
            ),
        ]
    )

    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"query": query, "chat_history": chat_history})
    return {"messages": [AIMessage(content=answer)]}


def grade_documents(state):
    """
    Determine whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["messages"][-1].content
    documents: list[Document] = state["documents"]

    # Score each doc
    filtered_docs = []

    formatted_blocks = "\n\n---\n\n".join(
        [
            f'Block {i + 1}:\n\n"""\n{text}\n"""'
            for i, text in enumerate([d.page_content for d in documents])
        ]
    )

    user_prompt = (
        f'Here is the query: "{question}"\n\n'
        "Here are the retrieved text blocks:\n"
        f"{formatted_blocks}\n\n"
        f"You should provide exactly {len(documents)} rankings, in order."
    )

    scores = GradeDocuments.model_validate(
        retrieval_grader.invoke(
            {
                "user_prompt": user_prompt,
            }
        )
    )

    for idx, score in enumerate(scores.block_rankings):
        if score.relevance_score >= 0.7:
            documents[idx].metadata["relevance_score"] = score.relevance_score
            documents[idx].metadata["reasoning"] = score.reasoning
            filtered_docs.append(documents[idx])

    return {"documents": filtered_docs}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    original_question = state["messages"][-1].content

    # Re-write question
    better_question = question_rewriter.invoke({"question": original_question})
    rewritten_message = HumanMessage(
        content=better_question,
        metadata={"is_rewritten": True, "original_question": original_question},
    )
    return {"messages": [rewritten_message]}


### Edges


def decide_to_retrieve(state):
    """
    Determine whether to generate an answer, or going through the RAG pipeline

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    question = state["messages"][-1].content
    routing = QueryRouting.model_validate(query_router.invoke({"question": question}))
    if routing.need_rag:
        return "need_rag"
    else:
        return "no_need_rag"


def decide_to_generate(state):
    """
    Determine whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if len(filtered_documents) < 3:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determine whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["messages"][-2].content  # skip AI generated message
    documents = state["documents"]
    generation = state["messages"][-1].content

    score = GradeHallucinations.model_validate(
        hallucination_grader.invoke({"documents": documents, "generation": generation})
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = GradeAnswer.model_validate(
            answer_grader.invoke({"question": question, "generation": generation})
        )
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("direct_generate", direct_generate)  # direct_generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    decide_to_retrieve,
    {"need_rag": "retrieve", "no_need_rag": "direct_generate"},
)
workflow.add_edge("direct_generate", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
graph = workflow.compile()
