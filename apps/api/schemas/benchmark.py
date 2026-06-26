from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Literal, Union
from enum import Enum
from datetime import datetime


class DatasetTypeEnum(str, Enum):
    medqa = "medqa"
    medmcqa = "medmcqa"
    mmlu = "mmlu"
    qa4mre = "qa4mre"
    all = "all"


class BenchmarkRunRequest(BaseModel):
    """Schema for triggering a benchmark run."""
    dataset: DatasetTypeEnum = Field(..., description="The target dataset to test, or 'all'.")
    n_calls: int = Field(default=1, description="Number of times to query the API per question.")
    concurrency: int = Field(default=10, description="Maximum concurrent API requests.")


# Define acceptable options
BenchmarkStatusType = Literal["pending", "started", "running", "completed", "failed"]


class BenchmarkState(BaseModel):
    """Schema for a benchmark run state"""
    model_config = ConfigDict(validate_assignment=True)

    run_id: str = Field(..., description="Unique identifier of the benchmark run.")
    dataset: DatasetTypeEnum = Field(..., description="The target dataset to test, or 'all'.")
    n_calls: int = Field(default=1, description="Number of times to query the API per question.")
    concurrency: int = Field(default=10, description="Maximum concurrent API requests.")
    benchmark_status: BenchmarkStatusType = Field(default="pending", description="Benchmark run status.")
    total_tasks: Union[int, Literal["Not calculated yet"]] = Field(
        default="Not calculated yet",
        description="Number of total tasks for the benchmark run. Accepts any integer, or exactly the string 'Not calculated yet'."
    )
    completed_tasks: int = Field(default=0, description="Number of tasks completed.")
    pending_tasks: Union[int, Literal["Not calculated yet"]] = Field(
        default="Not calculated yet",
        description="Number of pending tasks. Accepts any integer, or exactly the string 'Not calculated yet'."
    )
    progress_percent: float = Field(default=0.0, description="Progress percentage of the benchmark run.")
    started_at: datetime = Field(..., description="Starting datetime value of the benchmark.")
    jsonl_file: str | None = Field(..., description="JSONL file with intermediate results of the benchmark.")
    excel_file: str | None = Field(default=None, description="Excel file with consolidated results of the benchmark.")
    plot_files: list = Field(default=[], description="List of plot files of the benchmark.")
    error: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_completed_tasks_less_than_or_equal_to_total_tasks(self) -> "BenchmarkState":
        # Skip validation if total_tasks is the string "Not calculated yet"
        if isinstance(self.total_tasks, str):
            return self

        # Enforce that completed_tasks must not be greater than total_tasks
        if self.completed_tasks > self.total_tasks:
            raise ValueError("completed_tasks must be less than or equal to total_tasks")
            
        return self
    
    @model_validator(mode="after")
    def validate_pending_tasks_less_than_or_equal_to_total_tasks(self) -> "BenchmarkState":
        # Skip validation if total_tasks or pending_tasks is the string "Not calculated yet"
        if isinstance(self.total_tasks, str) or isinstance(self.pending_tasks, str):
            return self

        # Enforce that pending_tasks must not be greater than total_tasks
        if self.pending_tasks > self.total_tasks:
            raise ValueError("pending_tasks must be less than or equal to total_tasks")
            
        return self


class BenchmarkExportRequest(BaseModel):
    """Schema for consolidating a run's JSONL results into an Excel sheet."""
    run_id: str = Field(..., description="The unique benchmark run identifier to export.")
    custom_filename: str | None = Field(None, description="Optional custom destination filename (without extension) for the consolidated Excel file.")

class BenchmarkDeterministicRequest(BaseModel):
    """Schema for deterministic accuracy evaluation."""
    run_id: str = Field(..., description="The unique run identifier to evaluate.")
    attempt_id: int = Field(default=0, description="The specific attempt index to isolate.")
    dataset: DatasetTypeEnum = Field(default=DatasetTypeEnum.all, description="Specific dataset to plot, or 'all'.")
    custom_filename: str | None = Field(None, description="Optional custom destination file_name (without extension) fof for the output PNG plot.")
