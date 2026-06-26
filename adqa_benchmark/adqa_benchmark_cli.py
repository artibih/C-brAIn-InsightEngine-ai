import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import typer

from benchmark_logic import (
    DatasetType,
    PROCESSORS,
    get_deterministic_id,
    load_completed_ids,
    process_dataset,
    generate_deterministic_plot,
    generate_distributional_plot,
    analyze_telemetry_errors,
)

app = typer.Typer(name="benchmark-cli", help="Async CLI for robust LLM evaluation.")


@app.command()
def run(
    dataset: DatasetType = typer.Argument(
        ..., help="The target dataset to test, or 'all' to run the entire suite."
    ),
    source_path: Path = typer.Option(
        ..., "--source", "-s", 
        help="Path to the JSON file or directory containing the files."
    ),
    output_file: Path = typer.Option(
        Path("results_temp.jsonl"), "--output", "-o", 
        help="Path to the JSONL output file. Used for tracking state and resumption."
    ),
    n_calls: int = typer.Option(
        1, "--n-calls", "-n", 
        help="Number of times to query the API per question."
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", 
        help="Maximum number of concurrent API requests."
    ),
):
    """Evaluates the LLM asynchronously with resilient retry and resumption capabilities."""
    async def main():
        datasets_to_run = [d for d in DatasetType if d != DatasetType.all] if dataset == DatasetType.all else [dataset]
        completed_ids = load_completed_ids(output_file)

        # Pre-flight check: Load data and calculate pending tasks
        execution_plan = []
        total_pending_calls = 0

        for ds in datasets_to_run:
            target_file = source_path / f"{ds.value}.json" if dataset == DatasetType.all else source_path

            if not target_file.exists():
                typer.secho(f"[!] Warning: Expected file {target_file} not found.", fg=typer.colors.YELLOW)
                continue

            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                typer.secho(f"[!] Error: {target_file} is not valid JSON.", fg=typer.colors.RED)
                continue

            # Calculate exactly how many tasks are pending for this dataset
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

        if total_pending_calls == 0:
            typer.secho("All tasks have already been completed. Nothing to run.", fg=typer.colors.GREEN)
            return

        typer.echo(f"Initialization complete. Starting {total_pending_calls} pending API calls...\n")

        # Global Progress Bar Context
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            
            global_task = progress.add_task("[cyan]Overall Progress...", total=total_pending_calls)
            
            # Execute the plan
            for ds, data in execution_plan:
                # Standalone CLI queries the local HTTP endpoint (direct=False)
                await process_dataset(
                    dataset_name=ds,
                    data=data,
                    completed_ids=completed_ids,
                    out_path=output_file,
                    n_calls=n_calls,
                    concurrency_limit=concurrency,
                    progress_callback=lambda: progress.advance(global_task),
                    direct=False
                )

        typer.secho("\nEvaluation Run Completed Successfully.", fg=typer.colors.GREEN, bold=True)

    asyncio.run(main())


@app.command()
def info(
    dataset: DatasetType = typer.Argument(
        DatasetType.all, help="The target dataset to inspect, or 'all'."
    ),
    source_path: Path = typer.Option(
        ..., "--source", "-s", 
        help="Path to the JSON file or directory containing the files."
    )
):
    """Displays the total number of questions available in the specified dataset(s)."""
    console = Console()
    table = Table(title="Dataset Information")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Total Questions", justify="right", style="green")

    datasets_to_check = [d for d in DatasetType if d != DatasetType.all] if dataset == DatasetType.all else [dataset]

    for ds in datasets_to_check:
        target_file = source_path / f"{ds.value}.json" if dataset == DatasetType.all else source_path

        if not target_file.exists():
            table.add_row(ds.value, "[red]File not found[/red]")
            continue

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            table.add_row(ds.value, str(len(data)))
        except json.JSONDecodeError:
            table.add_row(ds.value, "[red]Invalid JSON[/red]")

    console.print(table)


@app.command()
def export(
    input_file: Path = typer.Option(
        Path("results_temp.jsonl"), "--input", "-i",
        help="Path to the JSONL results file to consolidate."
    ),
    output_file: Path = typer.Option(
        Path("results_consolidated.xlsx"), "--output", "-o",
        help="Path to the final Excel (.xlsx) file."
    )
):
    """Consolidates the temporary JSONL results into a single Excel file."""
    if not input_file.exists():
        typer.secho(f"[!] Error: Input file '{input_file}' does not exist.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Reading data from {input_file}...")

    try:
        df = pd.read_json(input_file, lines=True)
        if "id" in df.columns:
            df = df.sort_values(by="id")

        typer.echo(f"Exporting {len(df)} records to {output_file}...")
        df.to_excel(output_file, index=False, engine='openpyxl')
        typer.secho(f"Successfully exported data to {output_file}", fg=typer.colors.GREEN, bold=True)
        
        # Remove the temp file
        os.remove(input_file)

    except ValueError as e:
        typer.secho(f"[!] Data parsing error. Ensure the JSONL file is not empty and is valid. Details: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"[!] Export failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def deterministic(
    input_file: Path = typer.Option(
        Path("results_consolidated.xlsx"), "--input", "-i",
        help="Path to the consolidated Excel file to analyze."
    ),
    attempt_id: int = typer.Option(
        0, "--attempt-id", "-a",
        help="The specific attempt index (N) to isolate for the deterministic evaluation."
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.all, "--dataset", "-d",
        help="Select a specific dataset to plot, or 'all' to plot the full comparison."
    ),
    output_plot: Path = typer.Option(
        Path("deterministic_benchmark.png"), "--output-plot", "-p",
        help="Path to save the generated seaborn plot."
    )
):
    """Evaluates the deterministic accuracy (single trial) and benchmarks it visually against SOTA models."""
    try:
        generate_deterministic_plot(
            input_file=input_file,
            attempt_id=attempt_id,
            dataset=dataset,
            output_plot=output_plot
        )
        typer.secho(f"\nSuccess: Plot generated and saved to {output_plot}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"[!] Execution failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def distributional(
    input_file: Path = typer.Option(
        Path("results_consolidated.xlsx"), "--input", "-i",
        help="Path to the consolidated Excel file to analyze."
    ),
    dataset: DatasetType = typer.Option(
        DatasetType.all, "--dataset", "-d",
        help="Select a specific dataset to plot, or 'all' to plot the global benchmark."
    ),
    output_plot: Path = typer.Option(
        Path("distributional_benchmark.png"), "--output-plot", "-p",
        help="Path to save the generated seaborn plot."
    )
):
    """Evaluates the N-sample PMF metrics and benchmarks the system's uncertainty and consistency."""
    try:
        generate_distributional_plot(
            input_file=input_file,
            dataset=dataset,
            output_plot=output_plot
        )
        typer.secho(f"\nSuccess: Distributional plot generated and saved to {output_plot}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"[!] Execution failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("analyze-errors")
def analyze_errors(
    log_path: Path = typer.Option(
        Path("benchmark_errors.log"), "--log-path", "-l",
        help="Path to the error telemetry log file."
    ),
    output_plot: Path = typer.Option(
        Path("error_telemetry.png"), "--output-plot", "-p",
        help="Path to save the generated telemetry plot."
    )
):
    """Parses benchmark_errors.log and generates a plot visualizing transactional bottlenecks."""
    try:
        analyze_telemetry_errors(
            log_path=log_path,
            output_plot=output_plot
        )
        typer.secho(f"\nSuccess: Telemetry plot generated and saved to {output_plot}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"[!] Execution failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
