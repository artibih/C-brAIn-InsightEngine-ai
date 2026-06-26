import os
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from io import BytesIO

def read_csv_safely(path: str, nrows=None):
    """
    Reads CSV from:
    - Local filesystem
    - Azure Blob (abfs://)
    - Azure Blob (https://...blob.core.windows.net...)
    """

    if path.startswith("abfs://"):
        return pd.read_csv(
            path,
            nrows=nrows,
            storage_options={
                "account_name": os.getenv("AZURE_STORAGE_ACCOUNT"),
                "credential": DefaultAzureCredential(), 
            },
        )

    if "blob.core.windows.net" in path:

        credential = DefaultAzureCredential()

        blob_client = BlobClient.from_blob_url(
            blob_url=path,
            credential=credential
        )

        data = (
            blob_client
            .download_blob()
            .readall()
        )

        return pd.read_csv(
            BytesIO(data),
            nrows=nrows
        )

    return pd.read_csv(path, nrows=nrows)