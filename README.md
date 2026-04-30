# Personal Knowledge Garden

Your Private, Lifelong Multimodal Second Brain.

<img width="1919" height="1079" alt="Screenshot 2026-04-22 231141" src="https://github.com/user-attachments/assets/1af45ee8-cdb0-428c-9cb5-d5c9380d20a5" />


## What Changed

This version replaces the earlier single-file prototype with a cleaner FastAPI backend and a lightweight frontend built from pure HTML, Tailwind CSS, and vanilla JavaScript.

Most importantly, the indexing pipeline has been rebuilt to be robust and deterministic:

- Every supported file in `data/` is discovered through a manifest-driven sync pass.
- Every new or changed file is chunked and embedded explicitly with `nomic-embed-text:latest`.
- Uploaded files are indexed immediately after ingestion, so they become queryable right away.
- Manual file changes inside `data/` are picked up by sync checks before search, chat, and reflection operations.
- Code files are indexed with code-aware chunking that preserves structure and symbol hints.

## Features

- Persistent local chat sessions with create, rename, load, export, and delete.
- Garden Chat, Code Garden, and General Chat modes.
- Global semantic search across documents, images, and code.
- Code learn mode for summaries, documentation, refactoring ideas, and explanations.
- Related content suggestions for assistant answers.
- Auto-tagging during ingestion using `gemma3:4b`.
- Export chat as Markdown and export the full garden as ZIP.
- Code version history with diff support on re-upload.
- Backup and restore support.
- Settings panel for status, indexing controls, cache rebuild, and toggles.

## Requirements

- Python 3.11+
- Ollama running locally at `http://127.0.0.1:11434`
- Required models pulled locally:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text:latest
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn backend.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Project Structure

```text
textknowledge-garden/
├── backend/
│   ├── main.py
│   ├── config.py
│   └── services.py
├── frontend/
│   └── index.html
├── data/
├── db/
├── requirements.txt
├── README.md
└── run.sh
```

## Notes

- The frontend uses Tailwind via CDN because that was explicitly requested for this iteration.
- All model inference and embeddings stay local through Ollama.
- No cloud AI APIs are used.
