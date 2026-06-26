import os
import time
import logging
import asyncio
from urllib.parse import urlparse, unquote

from azure.storage.blob import BlobServiceClient

from config.settings import settings
from src.upload.process_file import process_local_file
from src.checkpoint.checkpoint_db import CheckpointDB
import structlog

logger = structlog.get_logger(__name__)

_blob_service = None

def get_blob_service():
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
    return _blob_service


db = CheckpointDB()

POLL_INTERVAL_SECONDS = 2      
MAX_RETRIES = 3
STALE_MINUTES = 5              
PROCESS_TIMEOUT_SECONDS = 1800





async def process_document(doc: dict):
    file_hash = doc["file_hash"]
    file_name = doc["file_name"]
    collection_name = doc["collection_name"]
    temp_path = None

    logger.info(
        "Processing claimed document",
        hostname=os.environ.get("HOSTNAME"),
        file_hash=file_hash,
        file_name=file_name,
        retry_count=doc.get("retry_count"),
    )

    try:
        blob_service = get_blob_service()

        blob_name = f"{file_hash}_{file_name}"
        blob_client = blob_service.get_blob_client(
            container="papers",
            blob=blob_name
        )

        blob_url = blob_client.url

        temp_path = f"/tmp/{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        await process_local_file(
            temp_path=temp_path,
            file_hash=file_hash,
            file_name=file_name,
            collection_name=collection_name
        )

        db.mark_embed_done(file_hash)
        logger.info("JOB COMPLETE", file_hash=file_hash, file_name=file_name)

    except Exception as e:
        logger.exception("Error processing document", file_hash=file_hash)
        try:
            db.mark_failed(file_hash, str(e))
        except Exception:
            logger.exception("Could not update failed status", file_hash=file_hash)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def run_worker():
    logger.info(
        "SQL worker started",
        hostname=os.environ.get("HOSTNAME"),
        max_retries=MAX_RETRIES,
        stale_minutes=STALE_MINUTES,
    )

    while True:
        try:
            doc = db.claim_next_document(
                max_retries=MAX_RETRIES,
                stale_minutes=STALE_MINUTES,
            )

            if doc is None:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            try:
                await asyncio.wait_for(
                    process_document(doc),
                    timeout=PROCESS_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error("Document processing timed out", file_hash=doc["file_hash"])
                try:
                    db.mark_failed(doc["file_hash"], "processing timed out")
                except Exception:
                    logger.exception("Could not mark timed-out doc failed")
            except Exception:
                logger.exception("Document processing failed", file_hash=doc["file_hash"])
                try:
                    db.mark_failed(doc["file_hash"], "unhandled worker error")
                except Exception:
                    logger.exception("Could not mark failed doc failed")

        except Exception:
            logger.exception("Worker loop error")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

STALENESS_POLL_INTERVAL_SECONDS = 180   


async def staleness_poller():
    logger.info(
        "Staleness poller started",
        interval_seconds=STALENESS_POLL_INTERVAL_SECONDS,
        max_retries=MAX_RETRIES,
        stale_minutes=STALE_MINUTES,
    )
    while True:
        try:
            requeued = await asyncio.to_thread(
                db.requeue_unhealthy_documents,
                MAX_RETRIES,
                STALE_MINUTES,
            )
            if requeued:
                logger.info("Requeued unhealthy documents", count=requeued)
        except Exception:
            logger.exception("Staleness poller error")
        await asyncio.sleep(STALENESS_POLL_INTERVAL_SECONDS)


async def _main():
    results = await asyncio.gather(
        run_worker(),
        staleness_poller(),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            logger.error("Background task exited with exception", error=str(r))


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
