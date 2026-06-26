import os
import json
from pathlib import Path

import structlog
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = structlog.get_logger(__name__)

HALLUCINATION_BLOB_CONTAINER = "uploads"
HALLUCINATION_BLOB_NAME = "experiment_id.json"


def _get_blob_service() -> BlobServiceClient:

    blob_url = os.getenv("AZURE_BLOB_URL")

    logger.info("Using blob url", blob_url=blob_url)


    if not blob_url:
        raise ValueError("AZURE_BLOB_URL is missing")

    credential = DefaultAzureCredential()

    logger.info(" ------- After credential -----")

    return BlobServiceClient(account_url=blob_url, credential=credential)


def upload_hallucination_result(data: dict) -> None:
    if data is None:
        raise ValueError("data must not be None")

    blob_service = _get_blob_service()

    logger.info("Using container", container=HALLUCINATION_BLOB_CONTAINER)

    client = (blob_service.get_container_client(HALLUCINATION_BLOB_CONTAINER))

    client.get_blob_client(HALLUCINATION_BLOB_NAME).upload_blob(json.dumps(data, indent=2),overwrite=True)


def clear_hallucination_result() -> None:

    blob_service = _get_blob_service()

    client = (blob_service.get_container_client(HALLUCINATION_BLOB_CONTAINER))

    try:

        client.get_blob_client(HALLUCINATION_BLOB_NAME).delete_blob()

    except Exception:
        pass


def download_hallucination_result() -> dict:

    blob_service = _get_blob_service()

    client = (blob_service.get_container_client(HALLUCINATION_BLOB_CONTAINER))

    blob_data = (client.get_blob_client(HALLUCINATION_BLOB_NAME).download_blob().readall())

    return json.loads(blob_data)

from apps.api.schemas.benchmark import (
    BenchmarkState
)

BENCHMARK_STATES_FOLDER = Path("benchmark_states")
BENCHMARK_RESULTS_FOLDER = Path("benchmark_results")


class BechmarkBlobProcessor:
    """Centralizes Blob operations"""
    def __init__(self):
        self.blob_service = _get_blob_service()
        self.container_client = self.blob_service.get_container_client(HALLUCINATION_BLOB_CONTAINER)

    def save_benchmark_state(self, state_data: BenchmarkState) -> None:
        """Serializes and uploads the execution state of a specific benchmark run."""

        blob_name: str = f"{BENCHMARK_STATES_FOLDER}/{state_data.run_id}.json"

        logger.info(f"Uploading '{blob_name}'")

        self.container_client \
        .get_blob_client(blob_name) \
        .upload_blob(state_data.model_dump_json(indent=2), overwrite=True)

    def get_benchmark_state(self, run_id: str) -> BenchmarkState | None:
        """Downloads and deserializes the execution state of a benchmark run."""
        blob_name: str = f"{BENCHMARK_STATES_FOLDER}/{run_id}.json"

        try:
            blob_data = self.container_client.get_blob_client(blob_name).download_blob().readall()
            return BenchmarkState.model_validate_json(blob_data)
        except Exception:
            return None 
        
    def delete_benchmark_state(self, run_id: str) -> None:
        """Deletes the execution state of a benchmark run and its related files"""
        benchmark_state = self.get_benchmark_state(run_id=run_id)

        related_files = [benchmark_state.jsonl_file, benchmark_state.excel_file] + benchmark_state.plot_files
        
        for related_file in related_files:
            if related_file:
                try:
                    blob_name: str = f"{BENCHMARK_RESULTS_FOLDER}/{related_file}"

                    blob_client = self.container_client.get_blob_client(blob_name)
                    blob_client.delete_blob()
                except Exception as e:
                    raise Exception(f"Could not delete '{blob_name}' from the Blob") from e

        try:
            blob_name: str = f"{BENCHMARK_STATES_FOLDER}/{run_id}.json"

            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
        except Exception as e:
            raise Exception(f"Could not delete benchmark state '{run_id}'") from e

    def list_benchmark_files(self, preffix: str) -> list[str]:
        """Lists all benchmark files in the Blob"""
        blob_list = self.container_client.list_blobs(name_starts_with=preffix)

        blob_names_list = [blob.name for blob in blob_list]
        
        return blob_names_list

    def deserialize_benchmark_states_info(self, run_blob_names_list: list[str]) -> dict[str, dict]:
        """
        Takes an iterable of Azure Blob properties, downloads their JSON content,
        and reconstructs the dictionary of all runs.
        """
        benchmark_states_data = {}
        
        for blob_name in run_blob_names_list:
            try:

                blob_client = self.container_client.get_blob_client(blob_name)
                
                blob_data = blob_client.download_blob().readall()
                run_state = json.loads(blob_data)

                benchmark_states_data[run_state["run_id"]] = run_state
                
            except Exception as e:
                print(f"Error deserializing blob {blob_name}: {e}")
                continue
                
        return benchmark_states_data

    def upload_benchmark_file(self, file_path: Path) -> None:
        """Uploads a benchmark file to the Blob"""
        if not file_path.exists():
            raise ValueError(f"File '{file_path}' not found.")

        try:
            blob_name: str = f"{BENCHMARK_RESULTS_FOLDER}/{file_path.name}"

            with open(file=file_path, mode="rb") as data:
                self.container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True
                )
        except Exception as e:
            raise Exception(f"Could not upload file '{file_path}' to the Blob") from e

    def download_benchmark_file(self, file_name: str, target_file_path: Path) -> Path:
        """Downloads a benchmark file from the Blob"""
        blob_name: str = f"{BENCHMARK_RESULTS_FOLDER}/{file_name}"
        try:
            with open(file=target_file_path, mode="wb") as download_file:
                download_file.write(self.container_client.download_blob(blob_name).readall())
        except Exception as e:
            raise Exception(f"Could not download file '{blob_name}' from the Blob") from e
        
    def list_all_files(self) -> list:
        """Lists all benchmark files in the Blob"""
        blob_list = self.container_client.list_blobs()

        blob_names_list = [blob.name for blob in blob_list]

        return blob_names_list

    def delete_file(self, blob_name: str) -> None:
        """Deletes one file"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
        except Exception as e:
            raise Exception(f"Could not delete blob '{blob_name}' from the Blob") from e
