import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_db.weaviate_sections import WeaviateSectionChunksDB
from src.vector_db.weaviate_tables import WeaviateTableDB
from src.vector_db.weaviate_images import WeaviateImageDB
from src.vector_db.weaviate_summaries import WeaviateSummaryDB
from src.vector_db.weaviate_metadata import WeaviateMetadataDB
import sqlite3

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def to_list(x):
    """Single embedding value to list of floats for Parquet."""
    if x is None or isinstance(x, dict): 
        return []
    if hasattr(x, "tolist"): 
        return x.tolist()
    return list(x) if isinstance(x, (list, tuple)) else []


def to_embedding_list(embeddings) -> list:
    """Normalize embeddings to list of lists for Parquet."""
    return [to_list(x) for x in embeddings]


def export_weaviate_collection(collection_name: str, db_instance, output_dir: Path) -> Dict[str, Any]:
    """Export one Weaviate collection to metadata + optional embeddings Parquet. Returns count and files."""
    try:
        collection = db_instance.collection
        data = []
        embeddings = []
        try:
            total = collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            total = None
        for obj in tqdm(
            collection.iterator(include_vector=True),
            total=total,
            desc=collection_name,
            unit="obj",
            dynamic_ncols=True,
        ):
            props = obj.properties.copy()
            vec = getattr(obj, "vector", None)
            if vec is not None:
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()), None)
                embeddings.append(np.array(vec) if vec is not None else None)
            else:
                embeddings.append(None)
            props["_weaviate_id"] = str(obj.uuid) if hasattr(obj, "uuid") else None
            data.append(props)

        if not data:
            logger.warning("No objects in collection: %s", collection_name)
            return {"count": 0, "files": []}

        df_metadata = pd.DataFrame(data)
        metadata_file = output_dir / f"{collection_name}_metadata.parquet"
        df_metadata.to_parquet(metadata_file, engine="pyarrow", index=False)

        embedding_file = None
        if any(e is not None for e in embeddings):
            df_embeddings = pd.DataFrame({
                "_weaviate_id": df_metadata["_weaviate_id"],
                "embedding": [e.tolist() if e is not None else None for e in embeddings],
            })
            embedding_file = output_dir / f"{collection_name}_embeddings.parquet"
            emb_list = to_embedding_list(df_embeddings["embedding"])
            table = pa.table({
                "_weaviate_id": df_embeddings["_weaviate_id"],
                "embedding": pa.array(emb_list, type=pa.list_(pa.float64())),
            })
            pq.write_table(table, embedding_file)

        return {
            "count": len(df_metadata),
            "files": [str(metadata_file), str(embedding_file)] if embedding_file else [str(metadata_file)],
        }
    except Exception as e:
        logger.error("Export collection failed: %s - %s", collection_name, e)
        raise


def export_checkpoint_db(checkpoint_db_path: str, output_dir: Path) -> Optional[str]:
    """Dump checkpoint SQLite to checkpoints.csv. Returns path or None if DB missing."""
    if not Path(checkpoint_db_path).exists():
        logger.warning("Checkpoint DB not found: %s", checkpoint_db_path)
        return None
    out = output_dir / "checkpoints.csv"
    with sqlite3.connect(checkpoint_db_path) as conn:
        pd.read_sql_query("SELECT * FROM documents", conn).to_csv(out, index=False)
    return str(out)


def main():
    now = pd.Timestamp.now()
    output_dir = Path(f"./backups/tabular_{now.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_configs = [
        ("SectionChunks", WeaviateSectionChunksDB),
        ("Torpor_Tables", WeaviateTableDB),
        ("Images", WeaviateImageDB),
        ("Summary", WeaviateSummaryDB),
        ("PaperMetadata", WeaviateMetadataDB),
    ]
    collections_exported = {
        name: export_weaviate_collection(name, db_class(), output_dir)
        for name, db_class in collection_configs
    }

    checkpoint_file = export_checkpoint_db("./data/checkpoints.db", output_dir)
    total_objects = sum(c["count"] for c in collections_exported.values())
    manifest = {
        "export_date": now.isoformat(),
        "collections": collections_exported,
        "checkpoint_db": checkpoint_file,
        "total_objects": total_objects,
    }
    with open(output_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
