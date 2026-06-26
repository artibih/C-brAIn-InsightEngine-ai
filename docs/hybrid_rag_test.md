# Local Test for Hybrid Rag

## Step 1: Start Neo4j and Weaviate

```bash
docker compose up -d
```

- **Neo4j** — Browser UI: http://localhost:7474, Bolt: `bolt://localhost:7687`  
  Login: `neo4j` / `password` (set in `docker-compose.yml` via `NEO4J_AUTH`).
- **Weaviate** — http://localhost:8080

## Step 2: Set Up Required env variables

I tested with an ordinary OpenAI api, but it should work with Azure's, although I did not test it
```env
# OpenAI (required for both agents GraphQueryAgent + HybridQueryAgent)
OPENAI_API_KEY=sk-...

# Neo4j (for GraphQueryAgent)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Weaviate (for RAG), this is local test!
WEAVIATE_URL=http://localhost:8080
WEAVIATE_CLASS=Document

# RAG API base URL (for HybridQueryAgent)
RAG_BASE_URL=http://localhost:8000
```
## Step 3: Load Graph Data into Neo4j

Run the graph extraction, you can do 1 single file just to test it

```bash
# Ensure you have 1 JSON file in src/graph/input/ with a "text" key,
# you could use one from the initial ~60 batch you guys initially shared in Slack
python -m src.graph.main # This will build and upload the graph to Neo4j
```

**Verify data in Neo4j:**
- Open http://localhost:7474
- Login: `neo4j` / `password`
- Run: `MATCH (n) RETURN n`, this will show you the graph

## Step 4: Start the ordinary RAG API

In one terminal:

```bash
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Step 5: Ingest At Least One Document into Weaviate

In another terminal:

```bash
# Create a test document, you could simple take the "text" from the json
# used in step 3, and create a .txt version of that, for simplicity
# Ingest it
curl -X POST http://localhost:8000/api/v1/rag/ingest -F "file=@your_txt_file.txt"
```

## Step 6: Run HybridQueryAgent

The script currently has a hardcoded question I used to test it with 
some random paper. You could modify the question to be related to the one
you uploaded, and then simply run it

```bash
python -m src.agents.hybrid_query_agent
```