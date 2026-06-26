import json
from typing import Any, Dict, List
import asyncio
import os
import pandas as pd
import numpy as np
import shutil
from fastapi.encoders import jsonable_encoder
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from config.settings import settings as config
from src.prompts.statistical_execution import (
    STATISTICAL_EXECUTION_PROMPT,
    REFINEMENT_PROMPT,
    USER_PROMPT,
)
from src.execution_sandbox.run_code import run_python_code, summarize_error
from azure.storage.blob import BlobServiceClient
from utils.load_csv_from_blob import read_csv_safely
import re
import structlog
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
logger = structlog.get_logger(__name__)

class StatisticalExecutor:
    """
    Parallel-safe statistical executor.

    Design principles:
    - dataset_schema is DERIVED LOCALLY (never stored in state)
    - async = orchestration + LLM
    - sync = pandas, filesystem, code execution (thread pool)
    """

    def __init__(
        self,
        experiment_id: str,
        step_id: str,
        llm_selection: ResolvedLlmSelection | None = None,
    ):
        self.llm = get_llm(
            temperature=0,
            workload="background",
            llm_selection=llm_selection,
        )
        self.system_prompt = ChatPromptTemplate.from_template(
            STATISTICAL_EXECUTION_PROMPT
        )
        self.user_prompt = ChatPromptTemplate.from_template(USER_PROMPT)
        self.experiment_id = experiment_id
        self.step_id = step_id

        logger.info(
            "StatisticalExecutor initialized | experiment_id=%s | step_id=%s | workload=background",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
        )

    async def execute(self, step: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "StatisticalExecutor.execute started | experiment_id=%s | step_id=%s | step_task=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            task=step.get("task"),
        )
        datasets = []
        if step.get("dataset_requirements") is None:
            datasets = step.get("datasets", [])
        else:   
            datasets =step["dataset_requirements"]
        if not datasets:
            raise RuntimeError("No datasets provided")

        dataset_id = datasets[0] 
        logger.info(
            "StatisticalExecutor selected dataset | experiment_id=%s | step_id=%s | dataset_id=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            dataset_id=dataset_id,
        )
        dataset = next((d for d in state["uploaded_datasets"] if d.get("id") == dataset_id), None)
        if not dataset or "path" not in dataset:
            raise RuntimeError(f"Dataset missing path: {dataset}")

        loop = asyncio.get_running_loop()


        df, dataset_schema = await loop.run_in_executor(
            None,
            self._load_dataset_and_schema,
            dataset,
        )
        logger.info(
            "Dataset loaded | experiment_id=%s | step_id=%s | rows=%s | columns=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            n_rows=dataset_schema.get("n_rows"),
            n_columns=dataset_schema.get("n_columns"),
        )
        MAX_CODE_ATTEMPTS = config.max_code_attempts

        previous_code = None
        last_error = None

        logger.info(
            "StatisticalExecutor retry budget | experiment_id=%s | step_id=%s | max_attempts=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            max_attempts=MAX_CODE_ATTEMPTS,
        )
        for attempt in range(1, MAX_CODE_ATTEMPTS + 1):
          
            logger.info(
                "Code generation attempt started | experiment_id=%s | step_id=%s | attempt=%s/%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
                attempt=attempt,
                max_attempts=MAX_CODE_ATTEMPTS,
            )

            python_code = await self._generate_code(
                dataset_schema=dataset_schema,
                analysis_step=step["task"],
                validation_criteria=state.get("validation_criteria", []),
                previous_code=previous_code,
                last_error=last_error,
                attempt_artifact_dir= os.path.join(config.artifact_storage_path, self.experiment_id, self.step_id), #attempt_artifact_dir,
                dependency_results=state.get("dependency_results", {}),
                
            )
            logger.info(
                "Code generation completed | experiment_id=%s | step_id=%s | attempt=%s | code_length=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
                attempt=attempt,
                code_length=len(python_code or ""),
            )

            result = await loop.run_in_executor(
                None,
                self._execute_code,
                step,
                state,
                df,
                dataset_schema,
                python_code,
                os.path.join(config.artifact_storage_path, self.experiment_id, self.step_id), #attempt_artifact_dir,
            )

            if result["code_stderr"]:
                logger.warning(
                    "Code execution failed | experiment_id=%s | step_id=%s | attempt=%s | error=%s",
                    experiment_id=self.experiment_id,
                    step_id=self.step_id,
                    attempt=attempt,
                    error=result["code_stderr"],
                )
                last_error = result["code_stderr"]
                previous_code = python_code
                continue

            return {
                "python_code": python_code,
                "generated_artifacts": result.get("generated_artifacts", []),
                "code_attempts": attempt,
                "code_stderr": None,
                "structured_results": result.get("structured_results", {}),
            }


        return {
            "python_code": previous_code,
            "code_attempts": MAX_CODE_ATTEMPTS,
            "code_stderr": last_error or "Failed after max retries",
        }


    def _load_dataset_and_schema(self, dataset: Dict[str, Any]):

        df = read_csv_safely(dataset["path"])

        dataset_schema = {
            "id": dataset.get("id", "primary_dataset"),
            "path": dataset["path"],
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        return df, dataset_schema

    def _execute_code(
        self,
        step: Dict[str, Any],
        state: Dict[str, Any],
        df: pd.DataFrame,
        dataset_schema: Dict[str, Any],
        python_code: str,
        attempt_artifact_dir: str,
    ) -> Dict[str, Any]:

        os.makedirs(
        attempt_artifact_dir,
        exist_ok=True
    )

        logger.info(
            "Created artifact directory | path=%s",
            attempt_artifact_dir
        )

        stdout, stderr, globals_after = run_python_code(
            python_code,
            globals_dict={
                "df": df,
                "pd": pd,
                "np": np,
                "dataset_schema": dataset_schema,
                "attempt_artifact_dir": attempt_artifact_dir,
                "dependency_results": state.get("dependency_results", {}),
            },
            return_globals=True,
        )

        structured_results = globals_after.get("structured_results")
        if structured_results is not None:
            structured_results = self._make_json_safe(structured_results)
            structured_results = jsonable_encoder(structured_results)
        if stderr:
            logger.warning(
                "Code execution stderr | experiment_id=%s | step_id=%s | error=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
                error=stderr,
            )
            return {
                "python_code": python_code,
                "code_stderr": summarize_error(stderr),
            }

        if structured_results is None and step.get("depends_on"):
            logger.info(
                "Code execution missing structured_results for dependent step | experiment_id=%s | step_id=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
            )
            return {
                "python_code": python_code,
                "code_stderr": "structured_results was not defined or populated",
            }


        paths = []
        if not stderr:
            logger.info(
                "Code execution succeeded, checking for artifacts | experiment_id=%s | step_id=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
            )
            paths = self._upload_artifacts_to_blob( self.step_id,
                os.path.join(config.artifact_storage_path, self.experiment_id, self.step_id)
            )

        
        def normalize_plot_key(filename: str) -> str:
            name = os.path.basename(filename)

            name = os.path.splitext(name)[0]

            name = re.sub(r"^(boxplot|roc|hist|plot|chart)_+", "", name, flags=re.IGNORECASE)

            name = re.sub(r"_\d+$", "", name)

            name = name.lower()

            name = re.sub(r"[^a-z0-9]", "", name)

            return name


        unique = {}
        for path in paths:
            key = normalize_plot_key(path)

            if key not in unique:
                unique[key] = path

        paths = list(unique.values())
        logger.info(
            "Artifacts uploaded | experiment_id=%s | step_id=%s | num_artifacts=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            num_artifacts=len(paths),
        )
        return {
            "python_code": python_code,
            "code_stderr": None,
            "structured_results": structured_results,
            "generated_artifacts": paths,
        }

    async def _generate_code(
        self,
        dataset_schema: Dict[str, Any],
        analysis_step: Any,
        validation_criteria: List[str],
        previous_code: str | None,
        last_error: str | None,
        attempt_artifact_dir: str,
        dependency_results: Dict[str, Any],
    ) -> str:

        if previous_code and last_error:
            logger.info(
                "Refining code after error | experiment_id=%s | step_id=%s | error=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
                error=last_error,
            )
            prompt = ChatPromptTemplate.from_template(REFINEMENT_PROMPT)
            vars_ = {
                "dataset_schema": dataset_schema,
                "analysis_step": analysis_step,
                "validation_criteria": validation_criteria,
                "previous_code": previous_code,
                "error": last_error,
            }
        else:
            logger.info(
                "Generating initial code | experiment_id=%s | step_id=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
            )
            prompt = ChatPromptTemplate.from_template(USER_PROMPT)
            vars_ = {
                "dataset_schema": dataset_schema,
                "analysis_step": analysis_step,
                "validation_criteria": validation_criteria,
            }

        messages = []
        messages.extend(
            self.system_prompt.format_messages(
                attempt_artifact_dir=attempt_artifact_dir,
                dependency_results=dependency_results,
            )
        )
        messages.extend(prompt.format_messages(**vars_))

        response = await self.llm.ainvoke(messages)

        raw_response = response.content or ""

        logger.info(
            "Code generation response received | experiment_id=%s | step_id=%s | response_length=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            response_length=len(raw_response),
        )

        logger.info(
            "Raw LLM response preview | experiment_id=%s | step_id=%s | preview=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            preview=raw_response[:300]
        )

        code = raw_response.strip()

        match = re.search(
            r"```(?:python)?\s*(.*?)```",
            code,
            re.DOTALL
        )

        if match:
            logger.info(
                "Markdown code fence detected and removed | experiment_id=%s | step_id=%s",
                experiment_id=self.experiment_id,
                step_id=self.step_id,
            )

            code = match.group(1).strip()

        logger.info(
            "Final executable code prepared | experiment_id=%s | step_id=%s | preview=%s",
            experiment_id=self.experiment_id,
            step_id=self.step_id,
            preview=code[:300]
        )

        return code
    

    def _upload_artifacts_to_blob(self, step_id, local_dir) -> List[str]:

        credential = DefaultAzureCredential()

        blob_url = os.getenv("AZURE_BLOB_URL")

        if not blob_url:
            raise ValueError("AZURE_BLOB_URL is missing")

        blob_service = BlobServiceClient(
        account_url=blob_url,
        credential=credential
        )

        container_name = os.getenv(
        "AZURE_BLOB_CONTAINER",
        "artifacts"
        )
        logger.info(
            "Uploading artifacts to blob storage | experiment_id=%s | step_id=%s | local_dir=%s",
            experiment_id=self.experiment_id,
            step_id=step_id,
            local_dir=local_dir,
        )
        container_client = blob_service.get_container_client(container_name)
        blob_paths = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                safe_name = self.sanitize_filename(file)
                logger.info("Uploading file: %s", file, experiment_id=self.experiment_id, step_id=step_id)
                logger.info("Sanitized filename: %s", safe_name, experiment_id=self.experiment_id, step_id=step_id)
                if safe_name != file:
                    src = os.path.join(root, file)
                    dst = os.path.join(root, safe_name)

                    if os.path.exists(dst):
                        base, ext = os.path.splitext(safe_name)
                        counter = 1
                        while True:
                            new_name = f"{base}_{counter}{ext}"
                            new_dst = os.path.join(root, new_name)
                            if not os.path.exists(new_dst):
                                dst = new_dst
                                safe_name = new_name
                                break
                            counter += 1

                    os.rename(src, dst)
                    
                local_path = os.path.join(root, safe_name)

                relative_path = os.path.relpath(local_path, local_dir)
                blob_path = f"{self.experiment_id}/{step_id}/{relative_path}"
                blob_paths.append(f"{blob_url}/{container_name}/{blob_path}")
                blob_client = container_client.get_blob_client(blob_path)

                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info("Uploaded file to blob storage | experiment_id=%s | step_id=%s | blob_path=%s", experiment_id=self.experiment_id, step_id=step_id, blob_path=blob_path)
        return blob_paths

    def sanitize_filename(self, name: str) -> str:

        base, ext = os.path.splitext(name)
        base = re.sub(r"[^A-Za-z0-9_]+", "_", base)
        base = re.sub(r"_+", "_", base)
        base = base.strip("_")

        if not base:
            base = "plot"

        return base + ext
    
        
    def _make_json_safe(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, pd.Series):
            return obj.tolist()

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, dict):
            
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, list):
                    k = ",".join(map(str, k))
                elif isinstance(k, tuple):
                    k = ",".join(map(str, k))
                else:
                    k = str(k)
                new_dict[k] = self._make_json_safe(v)
            return new_dict
        
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]

        return obj
