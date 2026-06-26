from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import json
import structlog
from pathlib import Path
from datetime import datetime
import pandas as pd

from adqa_benchmark.benchmark_logic import (
    DatasetType,
    PROCESSORS,
    get_deterministic_id,
    load_completed_ids,
    process_dataset,
    generate_deterministic_plot
)
from apps.api.schemas.benchmark import (
    DatasetTypeEnum,
    BenchmarkState,
    BenchmarkRunRequest,
    BenchmarkExportRequest,
    BenchmarkDeterministicRequest
)

from utils.blob_storage import (
    BechmarkBlobProcessor,
    BENCHMARK_STATES_FOLDER,
    BENCHMARK_RESULTS_FOLDER,
)

router = APIRouter()
logger = structlog.get_logger()


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SOURCE_DIR = WORKSPACE_ROOT / "adqa_benchmark" / "data_folder"
BENCHMARK_WORKSPACE = WORKSPACE_ROOT / "data" / "benchmarks"
BENCHMARK_WORKSPACE.mkdir(parents=True, exist_ok=True)



def calculate_progress_percent(total_tasks: int, completed_tasks: int) -> float:
    """Calculates progress percent"""

    progress_percent = round((completed_tasks / total_tasks) * 100, 2) if total_tasks > 0 else 0.0
    
    return progress_percent


async def async_run_benchmark(
    run_id: str,
    dataset: DatasetType,
    n_calls: int,
    concurrency: int,
    source_path: Path,
    output_file: Path
):
    """Asynchronous worker that runs evaluations in the background, updating progress."""
    benchmark_blob_processor = BechmarkBlobProcessor()

    benchmark_state = benchmark_blob_processor.get_benchmark_state(run_id=run_id)

    if benchmark_state.benchmark_status == "pending":
  
        benchmark_state.benchmark_status = "running"

        output_file.touch()
    
    if benchmark_state.benchmark_status == "failed":

        benchmark_state.benchmark_status = "running"

        benchmark_blob_processor.download_benchmark_file(
            blob_name=output_file.name,
            target_file_path=output_file
        )

    try:
        datasets_to_run = [d for d in DatasetType if d != DatasetType.all] if dataset == DatasetType.all else [dataset]
        completed_ids = load_completed_ids(output_file)

        execution_plan = []
        total_tasks = 0
        total_pending_calls = 0

        for ds in datasets_to_run:

            target_file = source_path / f"{ds.value}.json" if dataset == DatasetType.all else source_path

            if not target_file.exists():
                message = f"Dataset file '{target_file}' not found during benchmark startup"
                logger.exception(message, target_file=str(target_file))
                raise HTTPException(status_code=500, detail=message)

            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]

            processor = PROCESSORS[ds]
            ds_pending = 0
            for item in data:
                prompt, _, _, _ = processor(item)
                base_id = get_deterministic_id(item, prompt)
                for i in range(n_calls):
                    if f"{ds.value}_{base_id}_{i}" not in completed_ids:
                        ds_pending += 1

            total_pending_calls += ds_pending
            execution_plan.append((ds, data))

        total_tasks = len(completed_ids) + total_pending_calls

        benchmark_state.total_tasks = total_tasks
        benchmark_state.completed_tasks = len(completed_ids)
        benchmark_state.pending_tasks = total_pending_calls

        if total_pending_calls == 0:
            benchmark_state.benchmark_status = "completed"
            logger.info("Nothing to run, all attempts already completed.", run_id=run_id)
            return

        def increment_progress():
            benchmark_state.completed_tasks += 1
            benchmark_state.pending_tasks -= 1

            benchmark_state.progress_percent = calculate_progress_percent(
                total_tasks=benchmark_state.total_tasks,
                completed_tasks=benchmark_state.completed_tasks
            )

            if benchmark_state.completed_tasks % 10 == 0:
                benchmark_blob_processor.upload_benchmark_file(file_path=output_file)
                benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)

        for ds, data in execution_plan:
            await process_dataset(
                dataset_name=ds,
                data=data,
                completed_ids=completed_ids,
                out_path=output_file,
                n_calls=n_calls,
                concurrency_limit=concurrency,
                progress_callback=increment_progress,
                direct=True 
            )


        benchmark_state.benchmark_status = "completed"
        benchmark_blob_processor.upload_benchmark_file(file_path=output_file)
        benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)

        logger.info("Benchmark run finished successfully.", run_id=run_id)

    except Exception as e:
        benchmark_state.benchmark_status = "failed"
        benchmark_state.error = str(e)
        benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)
    
        logger.exception("Failed executing background benchmark run", run_id=run_id)



@router.get("/info")
async def get_info(dataset: DatasetTypeEnum = DatasetTypeEnum.all):
    """Displays total questions in the specified dataset(s)."""
    src_dir = DEFAULT_SOURCE_DIR
    datasets_to_check = [d for d in DatasetType if d != DatasetType.all] if dataset == DatasetTypeEnum.all else [DatasetType(dataset.value)]

    info_summary = {}
    for ds in datasets_to_check:
        target_file = src_dir / f"{ds.value}.json"

        if not target_file.exists():
            info_summary[ds.value] = {"status": "error", "message": f"File {target_file} not found"}
            continue

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            info_summary[ds.value] = {"status": "success", "total_questions": len(data)}
        except Exception as e:
            info_summary[ds.value] = {"status": "error", "message": f"Failed parsing JSON: {str(e)}"}

    return info_summary


@router.post("/run")
async def run_benchmark(request: BenchmarkRunRequest, background_tasks: BackgroundTasks):
    """Initiates an evaluation run asynchronously in the background."""
    benchmark_blob_processor = BechmarkBlobProcessor()

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    

    src_path = DEFAULT_SOURCE_DIR
    if request.dataset != DatasetTypeEnum.all:
        src_path = DEFAULT_SOURCE_DIR / f"{request.dataset.value}.json"

    output_file = BENCHMARK_WORKSPACE / f"results_{run_id}.jsonl"
    
    benchmark_state = BenchmarkState.model_validate({
        "run_id": run_id,
        "dataset": request.dataset.value,
        "n_calls": request.n_calls,
        "concurrency": request.concurrency,
        "jsonl_file": str(output_file.name),
        "started_at": datetime.now().isoformat()
    })

    benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)

    background_tasks.add_task(
        async_run_benchmark,
        run_id=run_id,
        dataset=DatasetType(request.dataset.value),
        n_calls=request.n_calls,
        concurrency=request.concurrency,
        source_path=src_path,
        output_file=output_file
    )

    return {
        "status": "started",
        "run_id": run_id,
        "message": "Benchmark run successfully initiated in the background.",
        "progress_endpoint": f"/api/v1/benchmark/status?run_id={run_id}"
    }


@router.get("/status")
async def get_status(run_id: str | None = None):
    """
    Returns benchmark states and their corresponding info and progress.
    
    Args:
        run_id (str | None): Run identifier for the benchmark the get status.
            Leave it empty to return all reference states that have been executed.
    """
    benchmark_blob_processor = BechmarkBlobProcessor()

    if run_id:
        benchmark_state = benchmark_blob_processor.get_benchmark_state(run_id)
        if not benchmark_state:
            raise HTTPException(status_code=404, detail=f"Benchmark run with ID '{run_id}' not found.")
        else:
            return benchmark_state

    else:
        # List all runs
        run_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_STATES_FOLDER}/")
        benchmark_states_data = benchmark_blob_processor.deserialize_benchmark_states_info(run_blob_paths_list)

        return {
            "active_runs_count": len(benchmark_states_data),
            "message": "No benchmark runs have been initiated yet." if not benchmark_states_data else f"{len(benchmark_states_data)} run(s) tracked.",
            "runs": benchmark_states_data
        }

@router.delete("/runs/")
def delete_runs(run_ids: list[str]):
    """Deletes benchmark states requested"""
 
    if not run_ids:
        return {"message": "No IDs provided. No items deleted."}
    
    benchmark_blob_processor = BechmarkBlobProcessor()

    run_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_STATES_FOLDER}/")
    benchmark_states_data = benchmark_blob_processor.deserialize_benchmark_states_info(run_blob_paths_list)
    benchmark_states_list = [run_id for run_id in benchmark_states_data.keys()]

    run_ids_set = set(benchmark_states_list)
    actual_run_ids = [run_id for run_id in run_ids if run_id in run_ids_set]

    for run_id in actual_run_ids:
        benchmark_blob_processor.delete_benchmark_state(run_id=run_id)
    
    return {
        "message": f"Deleted {len(actual_run_ids)} benchmark states.",
        "states_deleted": actual_run_ids
    }


@router.delete("/runs/all")
def delete_all_runs():
    benchmark_blob_processor = BechmarkBlobProcessor()

    run_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_STATES_FOLDER}/")
    benchmark_states_data = benchmark_blob_processor.deserialize_benchmark_states_info(run_blob_paths_list)
    benchmark_states_list = [run_id for run_id in benchmark_states_data.keys()]

    for run_id in benchmark_states_list:
        benchmark_blob_processor.delete_benchmark_state(run_id=run_id)
    
    return {
        "message": f"All benchmark states were successfully deleted. {len(benchmark_states_list)} benchmark states were deleted.",
        "states_deleted": benchmark_states_list
    }


@router.post("/export")
async def export_results(request: BenchmarkExportRequest):
    """Consolidates the temporary JSONL results of a run into a clean Excel file."""
    benchmark_blob_processor = BechmarkBlobProcessor()
    benchmark_state = benchmark_blob_processor.get_benchmark_state(request.run_id)

    input_file = BENCHMARK_WORKSPACE / f"results_{request.run_id}.jsonl"
    benchmark_blob_processor.download_benchmark_file(
        file_name=benchmark_state.jsonl_file,
        target_file_path=input_file
    )

    output_path = BENCHMARK_WORKSPACE / Path(f"{request.custom_filename}.xlsx") if request.custom_filename else BENCHMARK_WORKSPACE / f"results_{request.run_id}.xlsx"
    try:
        df = pd.read_json(input_file, lines=True)
        if "id" in df.columns:
            df = df.sort_values(by="id")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False, engine='openpyxl')
      
        os.remove(input_file)
    
        benchmark_state.excel_file = str(output_path.name)
        benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)
        benchmark_blob_processor.upload_benchmark_file(file_path=output_path)

        return {
            "status": "success",
            "run_id": request.run_id,
            "excel_file": str(output_path.name),
            "message": "Successfully consolidated results into Excel spreadsheet."
        }
    except Exception as e:
        logger.exception("Failed results consolidation", run_id=request.run_id)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/deterministic")
async def evaluate_deterministic(request: BenchmarkDeterministicRequest):
    """Calculates empirical single-attempt accuracy and renders competitive benchmarks."""
    benchmark_blob_processor = BechmarkBlobProcessor()
    benchmark_state = benchmark_blob_processor.get_benchmark_state(request.run_id)

    input_file = BENCHMARK_WORKSPACE / f"results_{request.run_id}.xlsx"
    benchmark_blob_processor.download_benchmark_file(
        file_name=benchmark_state.excel_file,
        target_file_path=input_file
    )

    output_plot = BENCHMARK_WORKSPACE / Path(f"{request.custom_filename}.png") if request.custom_filename else BENCHMARK_WORKSPACE / f"deterministic_{request.run_id}.png"

    try:
        summary = generate_deterministic_plot(
            input_file=input_file,
            attempt_id=request.attempt_id,
            dataset=DatasetType(request.dataset.value),
            output_plot=output_plot
        )

        summary["plot_path"] = str(Path(summary["plot_path"]).name)

        benchmark_state.plot_files.append(str(output_plot.name))
        benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)
        benchmark_blob_processor.upload_benchmark_file(file_path=output_plot)

        return summary
    except Exception as e:
        logger.exception("Deterministic analysis failed", run_id=request.run_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_all():
    """Lists all benchmark file paths in the blob"""
    benchmark_blob_processor = BechmarkBlobProcessor()
    run_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_STATES_FOLDER}/")
    results_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_RESULTS_FOLDER}/")

    file_paths = run_blob_paths_list + results_blob_paths_list
    
    return {"Benchmark file paths": file_paths}


@router.get("/download")
async def download_file(filename: str):
    """"Downloads a benchmark results file from the blob provided its filename"""

    target_filepath = BENCHMARK_WORKSPACE / filename 

    benchmark_blob_processor = BechmarkBlobProcessor()

    benchmark_blob_processor.download_benchmark_file(
        file_name=filename,
        target_file_path=target_filepath
    )

    return FileResponse(path=target_filepath, filename=filename)


@router.delete("/remove")
async def remove(filepath: Path):
    """Removes a file from the blob given its full path"""
    blob_name = str(filepath)
    file_name = filepath.name

    if filepath.suffix == ".json":
        raise HTTPException(status_code=400, detail="Cannot delete benchmark state (.json) file. Use the corresponding endpoint instead")
    try:
        benchmark_blob_processor = BechmarkBlobProcessor()

        run_blob_paths_list = benchmark_blob_processor.list_benchmark_files(preffix=f"{BENCHMARK_STATES_FOLDER}/")
        benchmark_states_data = benchmark_blob_processor.deserialize_benchmark_states_info(run_blob_paths_list)

        for benchmark_state in benchmark_states_data:
            removed_from_state = False   
            for file_type in ["jsonl_file", "excel_file"]:
                benchmark_file = benchmark_state[file_type]
                if file_name == benchmark_file:
                    benchmark_state[file_type] = None
                    removed_from_state = True
            if not removed_from_state:
                if file_name in benchmark_state.plot_files:
                    benchmark_state.plot_files.remove(file_name)

        benchmark_blob_processor.save_benchmark_state(state_data=benchmark_state)
        benchmark_blob_processor.delete_file(blob_name=blob_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "message": f"{filepath} successfully deleted."
    }
