import os
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import fitz


def read_pdf_safely(path: str, as_text=False):

    if path.startswith("abfs://"):

        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem(
            account_name=os.getenv("AZURE_STORAGE_ACCOUNT"),
            credential=DefaultAzureCredential()
        )

        with fs.open(path, "rb") as f:
            pdf_bytes = f.read()

    elif "blob.core.windows.net" in path:

        credential = DefaultAzureCredential()

        blob_service = BlobServiceClient(
            account_url=os.getenv("AZURE_BLOB_URL"),
            credential=credential
        )

        parsed = urlparse(path)

        parts = parsed.path.lstrip("/").split("/")

        container = parts[0]
        blob_name = "/".join(parts[1:])

        blob = (
            blob_service
            .get_container_client(container)
            .get_blob_client(blob_name)
        )

        pdf_bytes = blob.download_blob().readall()

    else:

        with open(path, "rb") as f:
            pdf_bytes = f.read()

    if as_text:
        return _extract_text_from_pdf(pdf_bytes)

    return pdf_bytes


def _extract_text_from_pdf(pdf_bytes):

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)