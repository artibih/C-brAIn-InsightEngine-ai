from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from config.settings import settings
from apps.api.routes import hypothesis, reviewer_three 
from apps.api.routes import rag
from apps.api.routes import graph
from apps.api.routes import benchmark
from apps.services.workflow_registry import graph_agent
from config.logging_config import (
    setup_logging
)

setup_logging()
logger = structlog.get_logger()

app = FastAPI(
    title="MVP2: Negative Data Analyzer",
    description="Agentic AI system for hypothesis testing and failure-mode discovery",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hypothesis.router, prefix="/api/v1/hypothesis", tags=["hypothesis"])
app.include_router(reviewer_three.router, prefix="/api/v1/reviewer-three", tags=["reviewer-three"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])  # NEW
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(benchmark.router, prefix="/api/v1/benchmark", tags=["benchmark"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting MVP2 API", environment=settings.environment)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MVP2 API")
    await graph_agent.close()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MVP2: Negative Data Analyzer",
        "version": "0.1.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment
    }