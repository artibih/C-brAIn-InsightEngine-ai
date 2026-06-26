import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.graph.ingestion import ingest_deterministic_graph, write_judge_edge
from src.graph.main import get_driver
from src.graph.tabular_export import (
    EXPORT_DIR as DEFAULT_EXPORT_DIR,
    count_export_papers,
    load_from_tabular,
)

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Neo4j from graph_export CSVs (papers, claims, experiments, judge_edges)."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=f"Directory with CSVs (default: env GRAPH_EXPORT_DIR or {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first paper-level ingestion failure (default: log and continue).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (tabular row skips, etc.).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    export_dir = (args.export_dir or Path(os.environ.get("GRAPH_EXPORT_DIR", DEFAULT_EXPORT_DIR))).resolve()
    if not export_dir.is_dir():
        print(f"Export directory not found: {export_dir}", file=sys.stderr)
        sys.exit(1)

    total_papers = count_export_papers(export_dir)
    if total_papers == 0:
        print(f"No papers found in {export_dir / 'papers.csv'}", file=sys.stderr)
        sys.exit(1)

    driver = get_driver()
    ok = 0
    failed = 0
    judge_ok = 0
    judge_fail = 0
    t0 = time.perf_counter()

    with driver.session() as session:
        for n, (paper_id, doi, name, content, judge_edges) in enumerate(
            load_from_tabular(export_dir), start=1
        ):
            elapsed = time.perf_counter() - t0
            avg = elapsed / n if n else 0.0
            eta = max(0.0, (total_papers - n) * avg)
            prefix = f"[{n}/{total_papers}] eta={_fmt_duration(eta)}"

            try:
                session.execute_write(ingest_deterministic_graph, content, paper_id, doi, name)
            except Exception as e:
                failed += 1
                logger.exception("%s FAILED paper_id=%s: %s", prefix, paper_id, e)
                if args.strict:
                    sys.exit(1)
                continue

            for result_id, claim_id, relationship, reason in judge_edges:
                try:
                    session.execute_write(
                        write_judge_edge, result_id, claim_id, relationship, reason
                    )
                    judge_ok += 1
                except Exception as e:
                    judge_fail += 1
                    logger.warning(
                        "%s judge edge failed paper_id=%s result_id=%s claim_id=%s: %s",
                        prefix,
                        paper_id,
                        result_id,
                        claim_id,
                        e,
                    )

            ok += 1
            logger.info(
                "%s OK paper_id=%s experiments=%s claims=%s judge_edges=%s",
                prefix,
                paper_id,
                len(content.experiments),
                len(content.claims),
                len(judge_edges),
            )

    elapsed = time.perf_counter() - t0
    print(
        f"Done in {_fmt_duration(elapsed)}. Papers ingested: {ok} ok, {failed} failed. "
        f"Judge edges: {judge_ok} ok, {judge_fail} failed."
    )
    if ok == 0 and total_papers > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
