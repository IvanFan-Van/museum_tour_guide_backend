from agent.utils import llm


def test_llm():
    res = llm.invoke("hello")
    assert res is not None
