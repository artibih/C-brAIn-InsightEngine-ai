import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)
for name in (
    "src.vector_db.weaviate_sections",
    "src.vector_db.weaviate_tables",
    "src.vector_db.weaviate_images",
    "src.vector_db.weaviate_metadata",
    "src.vector_db.weaviate_summaries",
    "src.vector_db.WeaviateDB",
    "sentence_transformers",
):
    logging.getLogger(name).setLevel(logging.WARNING)

from src.vector_db.backup_modules.export_to_tabular import (
    export_weaviate_collection,
    export_checkpoint_db,
)
from src.vector_db.weaviate_sections import WeaviateSectionChunksDB
from src.vector_db.weaviate_tables import WeaviateTableDB
from src.vector_db.weaviate_images import WeaviateImageDB
from src.vector_db.weaviate_summaries import WeaviateSummaryDB
from src.vector_db.weaviate_metadata import WeaviateMetadataDB
from src.embedding.mpnet_embedder import MpnetEmbedder


def tabular_export(output_dir: Path) -> None:
    """Export Weaviate collections + checkpoint DB to Parquet/CSV. Raises on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    collections_exported = {}
    shared_embedder = MpnetEmbedder()
    collection_configs = [
        ("SectionChunks", WeaviateSectionChunksDB, True),
        ("Torpor_Tables", WeaviateTableDB, True),
        ("Images", WeaviateImageDB, True),
        ("Summary", WeaviateSummaryDB, True),
        ("PaperMetadata", WeaviateMetadataDB, False),
    ]
    for name, db_class, use_embedder in tqdm(
        collection_configs,
        desc="Collections",
        unit="coll",
        dynamic_ncols=True,
    ):
        db = db_class(embedder=shared_embedder) if use_embedder else db_class()
        collections_exported[name] = export_weaviate_collection(name, db, output_dir)
    checkpoint_file = export_checkpoint_db("./data/checkpoints.db", output_dir)
    total_objects = sum(c["count"] for c in collections_exported.values())
    manifest = {
        "export_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "collections": {k: {"count": v["count"], "files": v.get("files", [])} for k, v in collections_exported.items()},
        "checkpoint_db": checkpoint_file,
        "total_objects": total_objects,
    }
    with open(output_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./backups/tabular_{timestamp}")
    tabular_export(output_dir)


if __name__ == "__main__":
    main()
