"""Restore Weaviate and optional checkpoint DB from a tabular (Parquet) backup."""

import sys
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.weaviate_sections import WeaviateSectionChunksDB
from src.vector_db.weaviate_tables import WeaviateTableDB
from src.vector_db.weaviate_images import WeaviateImageDB
from src.vector_db.weaviate_summaries import WeaviateSummaryDB
from src.vector_db.weaviate_metadata import WeaviateMetadataDB
from src.embedding.mpnet_embedder import MpnetEmbedder
import sqlite3

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def restore_weaviate_collection(
    collection_name: str,
    db_instance,
    backup_dir: Path,
    metadata_file: str,
    embeddings_file: Optional[str] = None,
) -> int:
    """Restore one Weaviate collection from metadata (and optional embeddings) Parquet. Returns count."""
    metadata_path = backup_dir / metadata_file
    if not metadata_path.exists():
        logger.warning("Metadata file not found: %s", metadata_path)
        return 0
    df_metadata = pd.read_parquet(metadata_path)

    embeddings_dict = {}
    if embeddings_file:
        emb_path = backup_dir / embeddings_file
        if emb_path.exists():
            df_emb = pd.read_parquet(emb_path)
            embeddings_dict = {
                row["_weaviate_id"]: np.array(row["embedding"])
                for _, row in df_emb.iterrows()
                if row.get("embedding") is not None and len(row["embedding"]) > 0
            }

    collection = db_instance.collection
    restored_count = 0
    for _, row in tqdm(
        df_metadata.iterrows(),
        total=len(df_metadata),
        desc=collection_name,
        unit="obj",
        dynamic_ncols=True,
    ):
        try:
            props = {k: v for k, v in row.items() if k != "_weaviate_id" and pd.notna(v)}
            embedding = embeddings_dict.get(row.get("_weaviate_id")) if embeddings_dict else None
            if embedding is not None:
                collection.data.insert(properties=props, vector=embedding.tolist())
            else:
                collection.data.insert(properties=props)
            restored_count += 1
        except Exception as e:
            logger.error("Restore object failed %s: %s", collection_name, e)
            continue
    logger.info("Restored %s: %s", collection_name, restored_count)
    return restored_count


def restore_checkpoint_db(checkpoint_path: Path, db_path: Path) -> None:
    """Overwrite checkpoint SQLite DB with CSV data."""
    if not checkpoint_path.exists():
        logger.warning("Checkpoint CSV not found: %s", checkpoint_path)
        return
    df = pd.read_csv(checkpoint_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, path TEXT NOT NULL, ocr_done INTEGER NOT NULL DEFAULT 0, embed_done INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("DELETE FROM documents")
        df.to_sql("documents", conn, if_exists="append", index=False)
        conn.commit()
    logger.info("Checkpoint DB restored %s: %s records", db_path, len(df))


def main():
    backup_dir = Path("./backups/tabular_20260217_220613")  # edit as needed
    skip_checkpoint = False
    if not backup_dir.exists():
        logger.error("Backup directory not found: %s", backup_dir)
        sys.exit(1)
    if not (backup_dir / "export_manifest.json").exists():
        logger.error("No export_manifest.json in %s", backup_dir)
        sys.exit(1)

    collection_configs = [
        ("SectionChunks", WeaviateSectionChunksDB, "SectionChunks_metadata.parquet", "SectionChunks_embeddings.parquet", True),
        ("Torpor_Tables", WeaviateTableDB, "Torpor_Tables_metadata.parquet", "Torpor_Tables_embeddings.parquet", True),
        ("Images", WeaviateImageDB, "Images_metadata.parquet", "Images_embeddings.parquet", True),
        ("Summary", WeaviateSummaryDB, "Summary_metadata.parquet", "Summary_embeddings.parquet", True),
        ("PaperMetadata", WeaviateMetadataDB, "PaperMetadata_metadata.parquet", "PaperMetadata_embeddings.parquet", False),
    ]
    shared_embedder = MpnetEmbedder()
    total_restored = 0
    try:
        for name, db_class, meta_file, emb_file, use_embedder in collection_configs:
            db = db_class(embedder=shared_embedder) if use_embedder else db_class()
            total_restored += restore_weaviate_collection(name, db, backup_dir, meta_file, emb_file)
    except Exception as e:
        logger.error("Weaviate restore failed: %s", e)
        raise

    if not skip_checkpoint:
        restore_checkpoint_db(backup_dir / "checkpoints.csv", Path("./data/checkpoints.db"))
    logger.info("Restoration done: %s objects", total_restored)


if __name__ == "__main__":
    main()
