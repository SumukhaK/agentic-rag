# Agentic RAG

A production-ready agentic retrieval-augmented generation system.

## Stack decisions so far

- **Language:** Python
- **Vector store:** Qdrant
- **Generation model:** Mistral / Mixtral, served locally via Ollama
- **Embedding model:** `nomic-embed-text`, served locally via Ollama
- **Evaluation:** Claude (Anthropic API) used as an evaluator/judge, not as the RAG generator
- **Agent orchestration:** custom agent loop, no framework (LangGraph/LlamaIndex not used)

## Status

Project scaffolding in progress. Architecture, API design, and repo structure still under discussion.
