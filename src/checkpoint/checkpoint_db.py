import os
import logging
import pyodbc
from datetime import datetime
from config.global_config import CONFIG
from src.document.base import Document
from config.settings import settings
logging.basicConfig(level=logging.INFO)


class CheckpointDB:
    def __init__(self):
        self.conn_str = self._build_connection_string()
        self._create_table()

  
    def _build_connection_string(self):
        return (
            f"DRIVER={{{settings.db_odbc_driver}}};"
            f"SERVER={settings.db_server};"
            f"DATABASE={settings.db_name};"
            f"UID={settings.db_user};"
            f"PWD={settings.db_password};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )

    def _get_conn(self):
        return pyodbc.connect(self.conn_str, timeout=30)

    def debug_documents(self, limit: int = 10):
        conn = self._get_conn()

        try:
            cursor = conn.cursor()
            limit = max(1, min(int(limit), 100))
            cursor.execute(f"""
                SELECT TOP {limit}
                    id,
                    blob_url,
                    file_name,
                    file_hash,
                    status,
                    retry_count,
                    error
                FROM documents
            """)

            rows = cursor.fetchall()

            for row in rows:
                print({
                    "id": row[0],
                    "blob_url": row[1],
                    "file_name": row[2],
                    "file_hash": row[3],
                })

        finally:
            conn.close()
            
    def _create_table(self):
        conn = self._get_conn()
        try:

            cursor = conn.cursor()
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='documents' AND xtype='U')
                CREATE TABLE documents (
                    id NVARCHAR(255),
                    path NVARCHAR(1024),
                    blob_url NVARCHAR(2048),
                    file_hash NVARCHAR(255) PRIMARY KEY,
                    file_name NVARCHAR(1024),
        
                    status NVARCHAR(50) NOT NULL DEFAULT 'queued',
                    retry_count INT NOT NULL DEFAULT 0,
                    ocr_done BIT NOT NULL DEFAULT 0,
                    embed_done BIT NOT NULL DEFAULT 0,

                    created_at DATETIME2,
                    started_at DATETIME2,
                    finished_at DATETIME2,

                    error NVARCHAR(MAX)
                )
            """)

            conn.commit()

        finally:
            conn.close()

    def add_document(self, doc: Document, file_hash: str, file_name: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            now = datetime.utcnow()

            try:
                cursor.execute("""
                    INSERT INTO documents (id, path, file_hash, file_name, status, created_at)
                    VALUES (?, ?, ?, ?, 'queued', ?)
                """, (doc.id, doc.original_path, file_hash, file_name, now))

                conn.commit()

            except pyodbc.IntegrityError:
                logging.info(f"Document already exists: {file_hash}")

        finally:
            conn.close()

    def add_pending_document(self, file_hash: str, file_name: str, collection_name: str):
        conn = self._get_conn()
        try: 
            cursor = conn.cursor()

            now = datetime.utcnow()

            try:
                cursor.execute("""
                    INSERT INTO documents (file_hash, file_name, status, created_at, collection_name)
                    VALUES (?, ?, 'pending_upload', ?, ?)
                """, (file_hash, file_name, now, collection_name))

                conn.commit()

            except pyodbc.IntegrityError:
                logging.warning(f"Already exists: {file_hash}")

        finally:
            conn.close()

    def update_status(self, file_hash: str, status: str, error: str = None):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET status = ?,
                    error = ?
                WHERE file_hash = ?
            """, (status, error, file_hash))

            conn.commit()
            
        finally:
            conn.close()

    def set_doc_id(self, file_hash: str, doc_id: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET id = ?
                WHERE file_hash = ?
            """, (doc_id, file_hash))

            conn.commit()
            
        finally:
            conn.close()

    def try_mark_processing(self, file_hash: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET status = 'processing',
                    started_at = GETUTCDATE()
                WHERE file_hash = ?
                AND status IN ('queued', 'failed')
            """, (file_hash,))

            conn.commit()
            success = cursor.rowcount == 1
        finally:    
            conn.close()

        return success

    def mark_completed(self, file_hash: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET status = 'completed',
                    retry_count = 0,
                    finished_at = GETUTCDATE(),
                    error = NULL
                WHERE file_hash = ?
            """, (file_hash,))

            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, file_hash: str, error: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET status = 'failed',
                    retry_count = retry_count + 1,
                    error = ?
                WHERE file_hash = ?
            """, (error, file_hash))

            conn.commit()
        finally:
            conn.close()
    
    def mark_queued(self, file_hash: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET status = 'queued',
                    error = NULL
                WHERE file_hash = ?
            """, (file_hash,))

            conn.commit()
        finally:
            conn.close()

    def mark_ocr_done(self, file_hash: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET ocr_done = 1
                WHERE file_hash = ?
            """, (file_hash,))

            conn.commit()
        finally:
            conn.close()

    def mark_embed_done(self, file_hash: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET embed_done = 1,
                    status='completed',
                    retry_count = 0,
                    finished_at = GETUTCDATE(),
                    error = NULL
                WHERE file_hash = ?
            """, (file_hash,))

            conn.commit()
        finally:
            conn.close()

    def is_completed(self, file_hash: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT TOP 1 1
                FROM documents
                WHERE file_hash = ?
                AND status = 'completed'
            """, (file_hash,))

            result = cursor.fetchone()
        finally:
            conn.close()

        return result is not None

    def hash_exists(self, file_hash: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT TOP 1 1
                FROM documents
                WHERE file_hash = ?
            """, (file_hash,))

            result = cursor.fetchone()
        finally:
            conn.close()

        return result is not None

    def document_path_exists(self, path: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT TOP 1 1
                FROM documents
                WHERE path = ?
            """, (path,))

            result = cursor.fetchone()
        finally:
            conn.close()

        return result is not None

    def set_blob_url(self, file_hash: str, blob_url: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE documents
                SET blob_url = ?
                WHERE file_hash = ?
            """, (blob_url, file_hash))

            conn.commit()
        finally:
            conn.close()

    def get_status(self, file_hash: str):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT status
                FROM documents
                WHERE file_hash = ?
            """, (file_hash,))

            row = cursor.fetchone()

        finally:
            conn.close()

        return row[0] if row else None
    
    def claim_next_document(self, max_retries: int = 3, stale_minutes: int = 30):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE TOP (1) documents WITH (ROWLOCK, READPAST, UPDLOCK)
                SET status = 'processing',
                    started_at = GETUTCDATE(),
                    retry_count = CASE
                        WHEN status = 'processing' THEN retry_count + 1
                        ELSE retry_count
                    END
                OUTPUT
                    inserted.file_hash,
                    inserted.blob_url,
                    inserted.file_name,
                    inserted.collection_name,
                    inserted.retry_count
                WHERE
                    (status IN ('queued', 'failed') AND retry_count < ?)
                    OR
                    (status = 'processing'
                    AND started_at < DATEADD(MINUTE, -?, GETUTCDATE())
                    AND retry_count < ?)
            """, (max_retries, stale_minutes, max_retries))

            row = cursor.fetchone()
            conn.commit()  # release the row lock; row is now safely 'processing'

            if row is None:
                return None

            return {
                "file_hash": row[0],
                "blob_url": row[1],
                "file_name": row[2],
                "collection_name": row[3],
                "retry_count": row[4],
            }
        finally:
            conn.close()
    
    def requeue_unhealthy_documents(self, max_retries: int = 3, stale_minutes: int = 30) -> int:
        """
        Bulk-recover rows that should be re-run, in one atomic statement:

        - 'processing' rows that have gone stale (worker crashed / hung past
            stale_minutes) -> requeue and count the attempt.
        - 'failed' rows still under the retry limit -> requeue.

        ROWLOCK + UPDLOCK + READPAST so this never blocks (and is never blocked by)
        workers claiming rows concurrently. Rows a worker is actively claiming are
        skipped via READPAST.

        Returns the number of rows requeued.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents WITH (ROWLOCK, READPAST, UPDLOCK)
                SET status = 'queued',
                    started_at = NULL,
                    retry_count = CASE
                        WHEN status = 'processing' THEN retry_count + 1
                        ELSE retry_count
                    END
                WHERE
                    (status = 'processing'
                    AND started_at < DATEADD(MINUTE, -?, GETUTCDATE())
                    AND retry_count < ?)
                    OR
                    (status = 'failed' AND retry_count < ?)
            """, (stale_minutes, max_retries, max_retries))
            requeued = cursor.rowcount
            conn.commit()
            return requeued
        finally:
            conn.close()

