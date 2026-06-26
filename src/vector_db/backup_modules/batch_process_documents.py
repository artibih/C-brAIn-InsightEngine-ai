import sys
import logging
import json
from pathlib import Path
from typing import List, Optional

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
    "src.processing.md_splitter",
    "src.vector_db.weaviate_sections",
    "src.vector_db.weaviate_tables",
    "src.vector_db.weaviate_images",
    "src.vector_db.weaviate_metadata",
    "src.vector_db.weaviate_summaries",
    "src.vector_db.WeaviateDB",
    "src.generation.paper_summary_generator",
    "src.generation.table_summary_generator",
    "src.generation.image_summary_generator",
    "src.generation.contextual_generator",
    "src.embedding.mistral_embedder",
    "sentence_transformers",
):
    logging.getLogger(name).setLevel(logging.WARNING)

from src.pipelines.rag_pipeline import BosnaRagPipeline
from src.document.base import Document

logger = logging.getLogger(__name__)


def load_json_document(json_path: Path) -> Optional[Document]:
    """Load JSON with 'text' key into Document. Optional: doi, paper_id, images/figures."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("text")
    if not text or not isinstance(text, str):
        logger.error("JSON file missing 'text' key or text is empty: file=%s", json_path)
        return None
    
    doi = data.get("doi")
    if doi and not isinstance(doi, str):
        doi = None
    
    paper_id = data.get("paper_id")
    if not paper_id:
        if doi:
            paper_id = doi
        else:
            paper_id = json_path.stem 
    
    metadata = {
        "paper_id": paper_id,
        "original_path": str(json_path),
    }
    
    if doi:
        metadata["doi"] = doi

    doc = Document(text=text, metadata=metadata)
    
    raw_images = data.get("images") or data.get("figures")
    if raw_images:
        doc.content["images"] = [img["fig_caption"] for img in raw_images if isinstance(img, dict) and img.get("fig_caption")]

    return doc


def process_batch(
    pipeline: BosnaRagPipeline,
    json_files: List[Path],
    start_index: int = 0,
    resume: bool = True,
) -> None:
    """Process json_files through pipeline."""
    pbar = tqdm(
        range(start_index, len(json_files)),
        desc="Documents",
        unit="doc",
        dynamic_ncols=True,
    )
    for i in pbar:
        json_file = json_files[i]
        file_path = str(json_file)
        if resume and pipeline.checkpoint_db.document_path_exists(file_path):
            continue
        try:
            doc = load_json_document(json_file)
            if doc is None:
                continue
            pipeline.storage.store_document(doc)
            pipeline.checkpoint_db.add_document(doc)
            pipeline.checkpoint_db.mark_ocr_done(doc)
            pipeline.add_document(doc)
        except Exception as e:
            logger.warning("Failed: %s - %s", file_path, e)


def main():
    folder_path = Path("data/input")
    start_index = 0
    resume = False

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Folder does not exist: path=%s", folder_path)
        sys.exit(1)
    pipeline = BosnaRagPipeline()
    json_files = sorted(folder.rglob("*.json"))
    json_files = json_files[:10]
    if not json_files:
        logger.warning("No JSON files found: folder=%s", folder_path)
        sys.exit(0)
    process_batch(
        pipeline=pipeline,
        json_files=json_files,
        start_index=start_index,
        resume=resume,
    )


if __name__ == "__main__":
    main()
