"""
Tabular export of extracted content and judge edges for cheap graph rebuilds.
Write CSVs (papers, claims, experiments, judge_edges) so the graph can be recreated
without re-running the costly LLM extraction and Judge.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.graph.schema.ontology import (
    Claim,
    Cohort,
    Experiment,
    ExtractedContent,
    Method,
    Result,
    ResultTrend,
)
from src.graph.ingestion import _scoped_id

logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).resolve().parent.parent.parent / "graph_export"

PAPERS_CSV = "papers.csv"
CLAIMS_CSV = "claims.csv"
EXPERIMENTS_CSV = "experiments.csv"
JUDGE_EDGES_CSV = "judge_edges.csv"

PAPERS_HEADER = ["paper_id", "doi", "name", "extracted_at", "source_paper_id"]
CLAIMS_HEADER = ["paper_id", "claim_id", "text", "status", "extracted_at"]
EXPERIMENTS_HEADER = [
    "paper_id",
    "experiment_id",
    "method_id",
    "method_name",
    "method_parameters",
    "cohort_id",
    "cohort_group_name",
    "cohort_species",
    "cohort_characteristics",
    "cohort_sample_size",
    "result_id",
    "result_description",
    "result_p_value",
    "result_trend",
    "extracted_at",
]
JUDGE_EDGES_HEADER = ["paper_id", "result_id", "claim_id", "relationship", "reason", "extracted_at"]


def ensure_export_dir(export_dir: Path) -> Path:
    export_dir = export_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def append_csv(path: Path, df: pd.DataFrame) -> None:
    """Append DataFrame to CSV, writing header only if file doesn't exist."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def extraction_timestamp() -> str:
    """Current UTC datetime in ISO format for extraction/export."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def export_paper(
    export_dir: Path,
    pid: str,
    doi: str | None,
    name: str | None = None,
    extracted_at: str | None = None,
    source_paper_id: str = "",
) -> None:
    export_dir = ensure_export_dir(export_dir)
    path = export_dir / PAPERS_CSV
    ts = extracted_at or extraction_timestamp()
    df = pd.DataFrame([{
        "paper_id": pid,
        "doi": doi or "",
        "name": name or "",
        "extracted_at": ts,
        "source_paper_id": source_paper_id,
    }])
    append_csv(path, df)


def export_claims(
    export_dir: Path,
    pid: str,
    content: ExtractedContent,
    extracted_at: str | None = None,
) -> None:
    export_dir = ensure_export_dir(export_dir)
    path = export_dir / CLAIMS_CSV
    ts = extracted_at or extraction_timestamp()
    if not content.claims:
        return
    df = pd.DataFrame([{
        "paper_id": pid,
        "claim_id": _scoped_id(pid, c.claim_id),
        "text": c.text,
        "status": c.status,
        "extracted_at": ts,
    } for c in content.claims])
    append_csv(path, df)


def export_experiments(
    export_dir: Path,
    pid: str,
    content: ExtractedContent,
    extracted_at: str | None = None,
) -> None:
    export_dir = ensure_export_dir(export_dir)
    path = export_dir / EXPERIMENTS_CSV
    ts = extracted_at or extraction_timestamp()
    if not content.experiments:
        return
    df = pd.DataFrame([{
        "paper_id": pid,
        "experiment_id": _scoped_id(pid, exp.experiment_id),
        "method_id": exp.method.stable_id,
        "method_name": exp.method.name,
        "method_parameters": exp.method.parameters or "",
        "cohort_id": exp.cohort.stable_id,
        "cohort_group_name": exp.cohort.group_name,
        "cohort_species": exp.cohort.species,
        "cohort_characteristics": exp.cohort.characteristics or "",
        "cohort_sample_size": exp.cohort.sample_size if exp.cohort.sample_size is not None else "",
        "result_id": exp.result.get_stable_id(_scoped_id(pid, exp.experiment_id)),
        "result_description": exp.result.description,
        "result_p_value": exp.result.p_value or "",
        "result_trend": exp.result.trend.value,
        "extracted_at": ts,
    } for exp in content.experiments])
    append_csv(path, df)

def export_judge_edges(
    export_dir: Path,
    pid: str,
    edges: list[tuple[str, str, str, str]],
    extracted_at: str | None = None,
) -> None:
    """Write judge edges. Each tuple is (result_id, claim_id, relationship, reason)."""
    if not edges:
        return
    export_dir = ensure_export_dir(export_dir)
    path = export_dir / JUDGE_EDGES_CSV
    ts = extracted_at or extraction_timestamp()
    df = pd.DataFrame([{
        "paper_id": pid,
        "result_id": result_id,
        "claim_id": claim_id,
        "relationship": relationship,
        "reason": reason or "",
        "extracted_at": ts,
    } for result_id, claim_id, relationship, reason in edges])
    append_csv(path, df)


def export_content(
    export_dir: Path,
    pid: str,
    doi: str | None,
    name: str | None,
    content: ExtractedContent,
    judge_edges: list[tuple[str, str, str, str]],
    *,
    source_paper_id: str = "",
) -> None:
    """Append one paper's data and its judge edges to the tabular export.
    """
    extracted_at = extraction_timestamp()
    export_paper(export_dir, pid, doi, name, extracted_at, source_paper_id=source_paper_id)
    export_claims(export_dir, pid, content, extracted_at)
    export_experiments(export_dir, pid, content, extracted_at)
    export_judge_edges(export_dir, pid, judge_edges, extracted_at)
    logger.debug("Exported paper %s to %s at %s", pid, export_dir, extracted_at)


def _safe_int(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _csv_str(value: object, default: str = "") -> str:
    """Coerce CSV/pandas cell to str; treat NaN/None as default."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() if value.strip() else default
    return str(value).strip() if str(value).strip() else default


def _csv_opt_str(value: object) -> str | None:
    s = _csv_str(value)
    return s if s else None


_CLAIM_STATUSES = frozenset({"Hypothesized", "Proven", "Refuted"})


def _claim_status(value: object) -> str:
    s = _csv_str(value, "Hypothesized")
    return s if s in _CLAIM_STATUSES else "Hypothesized"


def count_export_papers(export_dir: Path) -> int:
    """Number of distinct paper_id rows in papers.csv (for progress UI)."""
    export_dir = export_dir.resolve()
    papers_path = export_dir / PAPERS_CSV
    if not papers_path.exists():
        return 0
    df = pd.read_csv(papers_path)
    return int(df["paper_id"].nunique())


def _experiment_from_row(
    paper_id: str, row_index: int, r: pd.Series
) -> Experiment:
    """Build one Experiment from a CSV row (sanitized for NaN / empty cells)."""
    exp_id = _csv_str(r.get("experiment_id"), "")
    if not exp_id:
        exp_id = f"missing_exp_{paper_id}_{row_index}"

    method = Method(
        id=_csv_opt_str(r.get("method_id")),
        name=_csv_str(r.get("method_name"), "(unknown method)"),
        parameters=_csv_opt_str(r.get("method_parameters")),
    )
    cs = r.get("cohort_sample_size")
    sample_size: int | None = None
    try:
        if cs is not None and not pd.isna(cs):
            sample_size = _safe_int(_csv_str(cs, ""))
    except (TypeError, ValueError):
        sample_size = None

    cohort = Cohort(
        id=_csv_opt_str(r.get("cohort_id")),
        group_name=_csv_str(r.get("cohort_group_name"), "(unspecified cohort)"),
        species=_csv_str(r.get("cohort_species"), "Other"),
        characteristics=_csv_opt_str(r.get("cohort_characteristics")),
        sample_size=sample_size,
    )

    trend_str = _csv_str(r.get("result_trend"), "")
    try:
        trend = ResultTrend(trend_str) if trend_str else ResultTrend.INCONCLUSIVE
    except ValueError:
        trend = ResultTrend.INCONCLUSIVE

    result = Result(
        id=_csv_opt_str(r.get("result_id")),
        description=_csv_str(r.get("result_description"), "(no description)"),
        p_value=_csv_opt_str(r.get("result_p_value")),
        trend=trend,
    )
    return Experiment(
        experiment_id=exp_id,
        method=method,
        cohort=cohort,
        result=result,
    )


def load_from_tabular(export_dir: Path):
    """
    Read CSVs from export_dir and yield (paper_id, doi, content, judge_edges) per paper.
    judge_edges is a list of (result_id, claim_id, relationship, reason).
    """
    export_dir = export_dir.resolve()
    papers_path = export_dir / PAPERS_CSV
    claims_path = export_dir / CLAIMS_CSV
    experiments_path = export_dir / EXPERIMENTS_CSV
    judge_path = export_dir / JUDGE_EDGES_CSV
    
    if not papers_path.exists() or not experiments_path.exists():
        raise FileNotFoundError(f"Export dir must contain {PAPERS_CSV} and {EXPERIMENTS_CSV}: {export_dir}")

    papers_df = pd.read_csv(papers_path)
    claims_df = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame(columns=CLAIMS_HEADER)
    experiments_df = pd.read_csv(experiments_path)
    judge_df = pd.read_csv(judge_path) if judge_path.exists() else pd.DataFrame(columns=JUDGE_EDGES_HEADER)

    for paper_id in sorted(papers_df["paper_id"].unique()):
        paper_row = papers_df[papers_df["paper_id"] == paper_id].iloc[0]
        doi = paper_row.get("doi") if pd.notna(paper_row.get("doi")) else None
        name = paper_row.get("name") if pd.notna(paper_row.get("name")) else None

        paper_claims = claims_df[claims_df["paper_id"] == paper_id]
        claims: list[Claim] = []
        for ci, (_, row) in enumerate(paper_claims.iterrows()):
            try:
                claims.append(
                    Claim(
                        claim_id=_csv_str(row.get("claim_id"), f"unknown_claim_{paper_id}_{ci}"),
                        text=_csv_str(row.get("text"), "(empty claim)"),
                        status=_claim_status(row.get("status")),
                    )
                )
            except ValidationError as e:
                logger.warning(
                    "tabular_export: skip claim row | paper_id=%s row=%s error=%s",
                    paper_id,
                    ci,
                    e,
                )
        paper_experiments = experiments_df[experiments_df["paper_id"] == paper_id]
        experiments: list[Experiment] = []
        for ei, (_, r) in enumerate(paper_experiments.iterrows()):
            try:
                experiments.append(_experiment_from_row(paper_id, ei, r))
            except ValidationError as e:
                logger.warning(
                    "tabular_export: skip experiment row | paper_id=%s row=%s error=%s",
                    paper_id,
                    ei,
                    e,
                )

        content = ExtractedContent(experiments=experiments, claims=claims)

        paper_edges = judge_df[judge_df["paper_id"] == paper_id]
        edges: list[tuple[str, str, str, str]] = []
        for ji, (_, row) in enumerate(paper_edges.iterrows()):
            rid = _csv_str(row.get("result_id"))
            cid = _csv_str(row.get("claim_id"))
            if not rid or not cid:
                logger.warning(
                    "tabular_export: skip judge edge (missing result_id or claim_id) | paper_id=%s row=%s",
                    paper_id,
                    ji,
                )
                continue
            rel = _csv_str(row.get("relationship"), "").upper()
            if rel not in ("SUPPORTS", "CONTRADICTS", "INCONCLUSIVE"):
                logger.warning(
                    "tabular_export: skip judge edge (invalid relationship=%r) | paper_id=%s row=%s",
                    rel,
                    paper_id,
                    ji,
                )
                continue
            reason = _csv_str(row.get("reason"), "")
            edges.append((rid, cid, rel, reason))

        yield paper_id, doi, name, content, edges
