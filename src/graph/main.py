"""
Pipeline entry point: extract from text, ingest into Neo4j, run Judge.
Loads NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY from environment.

Reads JSON files from src/graph/input/ sequentially; each must have a "text" key. Optional "doi" is used as the Paper id when present.
Run from project root: python -m src.graph.main
    python -m src.graph.main               # incremental (MERGE)
    python -m src.graph.main --clean       # wipe graph first, then ingest
"""

import argparse
import json
import logging
import os
import atexit
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .extraction import extract_content, judge_and_link
from .ingestion import ingest_deterministic_graph
from .tabular_export import EXPORT_DIR as DEFAULT_EXPORT_DIR, export_content as export_content_tabular

logger = logging.getLogger(__name__)

INPUT_DIR = Path(__file__).resolve().parent / "input"


_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        atexit.register(_driver.close)
    return _driver


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def wipe_graph() -> int:
    """Delete all nodes and relationships from Neo4j. Returns the number of nodes deleted."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
        deleted = result.single()["deleted"]
    return deleted


def run_pipeline(
    text: str,
    paper_id: str,
    doi: str | None = None,
    export_dir: Path | None = None,
) -> None:
    """Extract content from text, ingest into Neo4j, run Judge, and optionally export to tabular."""
    title = extract_title(text)
    content = extract_content(text)
    driver = get_driver()
    pid = doi if doi else paper_id
    judge_edges: list[tuple[str, str, str, str]] = []
    with driver.session() as session:
        session.execute_write(ingest_deterministic_graph, content, paper_id, doi, title)
        judge_edges = session.execute_write(judge_and_link, content, pid)
    if export_dir is not None:
        export_content_tabular(
            export_dir, pid, doi, title, content, judge_edges, source_paper_id=paper_id
        )


def extract_title(text: str) -> str:
    """Extract paper title from first line of text. Removes leading '#' if present."""
    first_line = text.split("\n")[0].strip()
    return first_line.lstrip("#").strip() or "(no title)"


def load_paper_from_json(path: Path) -> tuple[str | None, str | None]:
    """Load JSON from path; return (text, doi). text is None if missing/invalid; doi is None if not present."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Skipping {path.name}: {e}")
        return None, None
    if not isinstance(data, dict):
        return None, None
    text = data.get("text")
    doi = data.get("doi") if isinstance(data.get("doi"), str) else None
    if not text or not isinstance(text, str):
        logger.warning("Skipping %s: no 'text' key or empty.", path.name)
        return None, doi
    return text, doi


def main():
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build knowledge graph from JSON papers")
    parser.add_argument("--clean", action="store_true", help="Wipe the entire Neo4j graph before ingesting")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0 = all)")
    args = parser.parse_args()

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        logger.warning("AZURE_OPENAI_API_KEY not set; extraction and Judge will fail.")
    if not os.environ.get("NEO4J_PASSWORD") and os.environ.get("NEO4J_URI"):
        logger.warning("NEO4J_PASSWORD not set; Neo4j may reject connection.")

    if not INPUT_DIR.is_dir():
        logger.error("Input directory not found: %s", INPUT_DIR)
        return

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files in %s", INPUT_DIR)
        return

    if args.clean:
        logger.info("--clean: wiping entire Neo4j graph...")
        deleted = wipe_graph()
        logger.info("Deleted %d nodes from Neo4j.", deleted)

    resume_from = os.environ.get("GRAPH_RESUME_FROM", "").strip()
    if resume_from:
        try:
            idx = next(i for i, p in enumerate(json_files) if p.stem == resume_from)
            json_files = json_files[idx:]
            logger.info("Resuming from %s.json (skipping %d papers)", resume_from, idx)
        except StopIteration:
            logger.warning("GRAPH_RESUME_FROM=%s not found in input; processing all", resume_from)

    if args.limit > 0:
        json_files = json_files[:args.limit]
        logger.info("Limiting to %d papers this run", args.limit)

    export_dir: Path | None = None
    if os.environ.get("GRAPH_EXPORT_DIR"):
        export_dir = Path(os.environ["GRAPH_EXPORT_DIR"]).resolve()
        logger.info("Tabular export enabled: %s", export_dir)
    else:
        export_dir = DEFAULT_EXPORT_DIR
        logger.info("Tabular export enabled (default): %s", export_dir)

    total = len(json_files)
    logger.info("Found %d JSON file(s) in %s/", total, INPUT_DIR.name)

    success, failed, skipped = 0, 0, 0
    batch_start = time.time()

    for i, path in enumerate(json_files, 1):
        paper_id = path.stem
        text, doi = load_paper_from_json(path)
        if text is None:
            skipped += 1
            logger.warning("[%d/%d] SKIP %s (no text)", i, total, path.name)
            continue

        logger.info("[%d/%d] Processing %s (doi=%s)", i, total, path.name, doi or "(none)")
        t0 = time.time()
        try:
            run_pipeline(text, paper_id=paper_id, doi=doi, export_dir=export_dir)
            success += 1
            status = "OK"
        except Exception:
            logger.exception("[%d/%d] FAILED %s", i, total, path.name)
            failed += 1
            status = "FAIL"

        file_elapsed = time.time() - t0
        batch_elapsed = time.time() - batch_start
        done = success + failed + skipped
        avg_per_file = batch_elapsed / done
        remaining_est = avg_per_file * (total - done)

        logger.info(
            "[%d/%d] %s in %.1fs | elapsed %s | avg %.1fs/paper | ETA %s | ok=%d fail=%d skip=%d",
            i, total, status, file_elapsed,
            _fmt_duration(batch_elapsed), avg_per_file,
            _fmt_duration(remaining_est), success, failed, skipped,
        )

    total_elapsed = time.time() - batch_start
    logger.info(
        "Finished in %s. Success: %d | Failed: %d | Skipped: %d | Total: %d",
        _fmt_duration(total_elapsed), success, failed, skipped, total,
    )


if __name__ == "__main__":
    main()
