GETTING START

Build pyproject environment
```bash
uv sync
```

For Windows, using the following command to activate environment
```bash
.venv/Script/activate
```

For Linux/Unix, using the following command to activate environment
```bash
source .venv/Script/activate
```

download spacy model first
```bash
uv add pip
uv run -- spacy download en_core_web_sm
```

Start Application
```bash
uvicorn src.main:app --reload
```

# Src Folder Structure

**main.py**: define API layer and endpoints, including 
- /api/v1/invoke
- /api/v1/health
- /

**models.py**: define websocket exchange data schema
- WSMessage
- WSTextMessage
- WSByteMessage
- WSStatusMessage
- WSTextChunkMessage
- WSStatusPayload
- WSTextChunkPayload
- WSQueryMessage
- WSQueryPayload

**graph.py**: Define LangGraph
- graph: graph object, can be called with .invoke() or .stream() or ...

**agent.py**: Agent Wrapper, used to convert specific agent implementation's output into unified output format.
- BaseAgent: Abstract class for all agent implementation
- LangGraphAgent: LangGraph Implemetation

**accumulator.py**: Define text to speech service, independent component
