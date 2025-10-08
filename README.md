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

