# GETTING START

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

# USEABLE COMMANDS

## zrok service related commands 
```bash
zrok2 agent status # check zrok service
zrok2 agent start # start zrok agent service
zrok2 agent stop # stop zrok agent service
```
you can check zrok agent console on address: "http://47.236.240.233:8889"

> Remind: zrok agent default listen to 8888 port. And nginx config at "./nginx.conf" to transport request that accesses 8889 to zrok console. Therefore, if you want to change access port. Please update nginx config and aliyun server port.

makesure /etc/nginx/nginx.conf set user to be innowing so that it can access to the ./frontend/dist folder
