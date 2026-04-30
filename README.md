# Personal Knowledge Garden

**A local-first multimodal AI knowledge system for private retrieval, code understanding, and persistent conversational reasoning**

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/74fd1dab-4b49-49c8-b1a9-e6ff8a01e5d1" />


## Overview

Personal Knowledge Garden is an advanced full-stack AI system that transforms a local folder of documents, images, and source code into a private, searchable, and conversational knowledge workspace. It is designed as a lifelong personal knowledge environment that runs fully on-device using local Ollama models, with no cloud AI dependency.

The project combines multimodal ingestion, persistent vector retrieval, code-aware reasoning, semantic search, session memory, backup workflows, and research-oriented retrieval tracing in a single local application. It was built to explore how far a practical, privacy-preserving retrieval-augmented generation system can go on low-resource hardware while still supporting professional-grade usability.

## Why This Project Is Strong

This project goes beyond a standard chatbot or document Q&A tool. It implements a complete local AI architecture with:

- multimodal ingestion for documents, images, and code
- persistent vector indexing with incremental synchronization
- code-aware chunking and code learning workflows
- local RAG over a personal knowledge base
- persistent chat sessions with export and restore support
- research-oriented retrieval tracing and evaluation hooks
- privacy-preserving on-device execution through Ollama

From a systems perspective, it is both a production-minded local application and a research-ready platform for efficient private RAG.

## Key Capabilities

- **Garden Chat**
  Retrieval-grounded chat over indexed local documents, notes, images, and code artifacts.

- **Code Garden**
  Specialized code analysis mode for summarization, explanation, documentation generation, refactoring guidance, and version comparison.

- **General Chat**
  Model-native chat mode for broader discussion outside the indexed knowledge garden.

- **Global Semantic Search**
  Natural-language search across all indexed local content with modality-aware filtering.

- **Persistent Session Memory**
  Named chat sessions with create, load, rename, delete, and Markdown export support.

- **Instant Local Indexing**
  New or updated files are synchronized into the index and become queryable immediately.

- **Version-Aware Code Storage**
  Re-uploaded code files maintain archived versions and support diff inspection.

- **Backup and Restore**
  Local garden state can be exported and restored for resilience and portability.

- **Research Traceability**
  Retrieval route, selected evidence, latency, verification, and benchmark evaluation hooks are exposed for analysis.

## Research-Ready Extension

The system now includes a lightweight research-oriented retrieval architecture called **BASIL-RAG**:

**Budget-Aware Structure-Indexed Local Retrieval-Augmented Generation**

BASIL-RAG introduces several advanced ideas while remaining practical on a low-spec laptop:

- **dual-granularity indexing**
  - file-memory summaries for efficient routing
  - chunk-memory embeddings for detailed retrieval

- **modality-aware routing**
  - document
  - code
  - image
  - hybrid
  - general

- **budget-aware context packing**
  - selects evidence under a fixed context budget
  - balances relevance, structure, redundancy, and freshness

- **lightweight answer verification**
  - runs a second local pass to reduce unsupported claims

- **evaluation logging**
  - logs route, retrieval latency, generation latency, selected evidence, and verification state

This makes the project suitable not just as an engineering portfolio piece, but also as the foundation for a publishable local RAG research system.

## System Architecture

The application follows a clean local-first architecture:

1. **Frontend Layer**
   A single-page interface built with HTML, Tailwind CSS, and vanilla JavaScript.

2. **API Layer**
   FastAPI backend exposing endpoints for health, indexing, ingestion, search, chat, code learning, backup/restore, and research evaluation.

3. **Service Layer**
   Central orchestration in `backend/services.py` for ingestion, chunking, indexing, retrieval, generation, verification, and persistence.

4. **AI Runtime Layer**
   Local Ollama integration for:
   - `gemma3:4b` for generation and verification
   - `nomic-embed-text:latest` for embeddings

5. **Storage Layer**
   - Chroma for persistent vector storage
   - JSON-backed manifest and session storage
   - local file system for source data, backups, and archived versions

## How It Works

### Ingestion

The system accepts documents, images, and code files from the local machine. Files are stored under the local `data/` directory and classified by modality.

### Processing

Each modality is processed differently:

- **documents**
  text extraction and section-aware chunking

- **images**
  metadata and description-based pseudo-text representation for retrieval

- **code**
  structure-preserving chunking with function, class, and comment-aware analysis

### Indexing

The backend creates:

- file-level semantic memory
- chunk-level semantic memory

Both are embedded locally using `nomic-embed-text:latest` and stored in persistent Chroma collections.

### Retrieval

At query time, the system:

1. embeds the user query
2. predicts the best retrieval route
3. gathers file and chunk candidates
4. packs evidence under a context budget
5. generates a grounded answer with `gemma3:4b`
6. optionally verifies the answer against retrieved evidence

### Persistence

Chats, file metadata, versions, backups, and research logs all remain on the local machine.

## Technology Stack

- **Backend:** FastAPI
- **Frontend:** HTML, Tailwind CSS, vanilla JavaScript
- **LLM Runtime:** Ollama
- **Primary Model:** `gemma3:4b`
- **Embedding Model:** `nomic-embed-text:latest`
- **Vector Store:** Chroma
- **PDF Parsing:** `pypdf`
- **Image Handling:** Pillow
- **Persistence:** local filesystem + JSON + persistent Chroma collections

## Engineering Highlights

This project demonstrates practical experience with:

- local LLM application engineering
- retrieval-augmented generation design
- multimodal indexing pipelines
- code-aware semantic retrieval
- vector database integration
- low-resource inference constraints
- incremental synchronization and cache-safe indexing
- full-stack product design with operational tooling
- research instrumentation for evaluation and benchmarking

## Project Structure

```text
Personal Knowledge Garden/
|-- backend/
|   |-- config.py
|   |-- main.py
|   `-- services.py
|-- frontend/
|   `-- index.html
|-- data/
|-- db/
|-- requirements.txt
|-- README.md
`-- run.sh
```

## Requirements

- Python 3.11+
- Ollama running locally at `http://127.0.0.1:11434`
- locally installed models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text:latest
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Open the app:

```text
http://127.0.0.1:8000
```

## Best Fit Use Cases

- private AI knowledge assistants
- local RAG systems
- code intelligence and developer learning tools
- personal research archives
- lifelong knowledge management systems
- privacy-preserving AI products

## License / Usage

This project is intended as a local AI systems project and advanced engineering portfolio piece. If you use it publicly, make sure local model usage and dependency licenses are reviewed for your target distribution.
