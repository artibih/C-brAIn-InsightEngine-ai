# InsightEngine AI service
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

A research-oriented AI platform for evidence-backed hypothesis testing, literature retrieval, and scientific reasoning over biomedical content.

This repository is part of a larger multi-service system composed of 4 core repositories that together form the full InsightEngine platform.

## System Context

The InsightEngine platform is a modular distributed system for biomedical research and AI-assisted scientific reasoning. It consists of four core repositories:

- **User Service (`C-brAIn-InsightEngine-user-service`)** — authentication, authorization, and user profile management
- **Backend API Service (`C-brAIn-InsightEngine-api`)** — core application backend, business logic, and system coordination layer
- **Web Frontend (`C-brAIn-InsightEngine-web`)** — user interface for interacting with the platform and visualizing results
- **AI Service (`C-brAIn-InsightEngine-ai` - this repository)** — orchestration engine for hypothesis evaluation, retrieval-augmented generation, knowledge graph reasoning, and scientific claim validation

Together, these components form an end-to-end system.

## Overview

`C-brAIn-InsightEngine-ai` combines FastAPI services, LangGraph-based orchestration, vector search, and graph-based exploration to help researchers investigate hypotheses, validate claims, and surface potential failure modes before committing to costly experiments.

The platform is designed for:
- biomedical and scientific research workflows
- literature-grounded question answering
- claim verification and hallucination detection
- dataset-aware analysis pipelines
- interactive graph exploration over structured knowledge

## Governance and Origin

This project was initiated and funded through the Consortium for Biomedical Research & AI in Neurodegeneration (C-BRAIN) and developed as a collaborative effort between Arti Analytics Inc. and 387Labs.

The system is part of a larger multi-repository platform (4 repositories in total) designed to support biomedical research workflows, including literature retrieval, knowledge graph reasoning, and AI-assisted hypothesis validation.

Development has been carried out through joint engineering and research efforts across partner teams, including contributions and feedback from collaborating academic researchers and affiliated research institutions. The focus has been on reproducibility, modular system design, and evidence-based scientific reasoning.

## Why this project matters

The repository implements a multi-stage pipeline that:
1. plans a research workflow,
2. retrieves relevant literature and structured context,
3. analyzes datasets when provided,
4. synthesizes findings,
5. extracts claims,
6. checks those claims against evidence, and
7. iterates on critique feedback.

This makes the system useful for teams working at the intersection of AI, knowledge retrieval, and scientific validation.

## Key Features

- **Agentic hypothesis evaluation** using a LangGraph workflow
- **Retrieval-augmented generation** for scientific documents
- **Optional statistical analysis** for user-supplied datasets
- **Claim extraction and hallucination detection** for evidence checking
- **Critique-and-revision loop** for improving outputs
- **Knowledge graph exploration** over connected biomedical entities and relationships
- **REST API** with interactive docs via FastAPI
- **Dockerized deployment** for local and small-team environments

## Architecture

The project is organized around three main layers:

### 1. API Layer
The FastAPI application exposes endpoints for:
- hypothesis evaluation streaming
- reviewer workflows
- RAG retrieval and document upload
- graph search and subgraph retrieval
- benchmark utilities

### 2. Orchestration Layer
The core workflow is defined in [src/orchestrator/graph.py](src/orchestrator/graph.py) and uses LangGraph to coordinate agents such as:
- `hypothesis_planner`
- `knowledge_retriever`
- `statistical_executor`
- `synthesizer`
- `claim_extractor`
- `semantic_arbiter`
- `hallucination_detector`
- `critique_agent`

### 3. Data & Retrieval Layer
The platform integrates with:
- **Weaviate** for vector retrieval
- **Neo4j** for graph queries and relationship exploration
- **Azure / OpenAI-compatible models** for embeddings and LLM inference
- local checkpointing and artifact storage

## Repository Structure

- [apps/api](apps/api) — FastAPI server and route definitions
- [apps/services](apps/services) — service integrations and workflow wiring
- [config](config) — settings and runtime configuration
- [src/agents](src/agents) — agent implementations
- [src/orchestrator](src/orchestrator) — workflow definitions
- [src/pipelines](src/pipelines) — document ingestion and RAG pipeline logic
- [src/vector_db](src/vector_db) — database integrations
- [utils](utils) — supporting utilities and processing scripts

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to one or more supported model providers (for example Azure OpenAI or OpenAI-compatible endpoints)
- Optional infrastructure:
  - Weaviate
  - Neo4j
  - Azure Blob Storage / queue services for document processing

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd C-brAIn-InsightEngine-ai
```

### 2. Configure environment variables

Copy the example environment file and update values for your deployment:

```bash
cp .env.example .env
```

At minimum, configure:
- API keys for your LLM provider(s)
- Azure OpenAI settings if you use Azure deployments
- Weaviate endpoint settings
- Neo4j credentials

### 3. Start services with Docker

```bash
docker compose up --build
```

This will start:
- the API service on port `8000`
- Neo4j on ports `7474` and `7687`
- Weaviate for vector search

## Usage

Once the API is running, the docs are available at:

```text
http://localhost:8000/docs
```

### Health check

```bash
curl http://localhost:8000/health
```

### Hypothesis evaluation example

```bash
curl -X POST "http://localhost:8000/api/v1/hypothesis/test/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "hypothesis": "Gene X is associated with increased risk of disease Y",
    "dataset_paths": [],
    "feedback": ""
  }'
```

### RAG retrieval example

```bash
curl -X GET "http://localhost:8000/api/v1/rag/retrieve/v5?query=association%20between%20insulin%20resistance%20and%20Alzheimer's%20disease"
```

### Graph API example

```bash
curl "http://localhost:8000/api/v1/graph/search?q=Alzheimer&limit=10"
```

## Configuration

The primary configuration entry point is [config/settings.py](config/settings.py). Runtime values are loaded from `.env` using Pydantic settings.

Important configuration areas include:
- LLM provider and deployment settings
- Weaviate connection parameters
- Neo4j connection parameters
- API host/port and logging behavior
- artifact and checkpoint storage paths

The repository also includes [config/rag_config.yaml](config/rag_config.yaml), which defines RAG-related settings and collection names.

## Deployment

### Local container deployment

The repository is designed to run with Docker Compose using the stack defined in [docker-compose.yml](docker-compose.yml).

### Cloud deployment notes

The application can be deployed to Azure or similar cloud environments by:
- containerizing the API and worker services,
- providing environment variables for the configured model providers,
- ensuring connectivity to Weaviate and Neo4j,
- mounting or configuring persistent storage for artifacts and checkpoints.

## Development

### Local development workflow

If you want to run the API without Docker:

```bash
uvicorn apps.api.main:app --reload
```

Make sure the data directory exists before running locally:

```bash
mkdir -p data
```

### Key development areas

- [apps/api](apps/api) — API routes and request handling
- [src/orchestrator](src/orchestrator) — workflow logic and graph behavior
- [src/agents](src/agents) — agent implementations
- [src/pipelines](src/pipelines) — ingestion and retrieval flow
- [config](config) — environment and runtime configuration

### Coding conventions

- Keep configuration externalized via environment variables where possible
- Prefer explicit, typed request/response schemas for API endpoints
- Document new workflows and runtime assumptions clearly

## Testing

This repository does not yet include a formal automated test suite.

Validation is currently performed through:
- FastAPI interactive documentation (Swagger UI)
- Manual endpoint testing
- Workflow-level integration checks across retrieval and graph components

## Contributing

Contributions are welcome.

A good contribution workflow is:
1. fork the repository
2. create a feature branch
3. make focused changes with clear documentation
4. run relevant local validation steps
5. open a pull request with a concise summary and testing notes

Please keep changes well-scoped and document any new configuration requirements.

## License

This project is licensed under the Apache License 2.0.

See the LICENSE file for the full license text.

## Support and next steps

For developers evaluating the project, the fastest path is:
1. configure the environment file,
2. start the Docker stack,
3. open the Swagger UI,
4. try the hypothesis and retrieval endpoints.

## Support

For issues, questions, or feature requests, please open a GitHub issue or contact the maintainers through the repository's preferred communication channel.

