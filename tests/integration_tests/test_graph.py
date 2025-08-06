import pytest

from agent import graph

pytestmark = pytest.mark.anyio


# @pytest.mark.langsmith
# async def test_agent_simple_passthrough() -> None:
#     inputs = {"question": "How's the porcelain made of?"}
#     res = await graph.ainvoke(inputs)
#     assert res is not None
