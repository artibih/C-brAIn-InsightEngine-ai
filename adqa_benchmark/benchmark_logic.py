import os
import string
import json
import asyncio
import hashlib
import aiohttp
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Telemetry & Logging Setup ---
logger = logging.getLogger("benchmark_error_logger")
logger.setLevel(logging.ERROR)

# Ensure it writes locally to a dedicated file in the directory
# When executed from apps/api, let's keep the error log path consistent
LOG_FILE_PATH = Path(__file__).parent / "benchmark_errors.log"
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Constants ---
API_URL = "http://localhost:8000/api/v1/rag/retrieve/v5"

# --- Static Benchmark Matrix ---
BENCHMARK_DATA = [
    {"Model": "ChatDoctor-7B", "medqa": 25.7, "medmcqa": 36.4, "mmlu": 46.9, "qa4mre": 51.4, "avg": 40.1},
    {"Model": "Med-Alpaca-7B", "medqa": 41.4, "medmcqa": 42.8, "mmlu": 44.9, "qa4mre": 57.1, "avg": 46.5},
    {"Model": "BiomedGPT-7B", "medqa": 38.8, "medmcqa": 41.9, "mmlu": 48.9, "qa4mre": 42.6, "avg": 43.1},
    {"Model": "Meditron-7B", "medqa": 27.6, "medmcqa": 31.4, "mmlu": 36.7, "qa4mre": 25.7, "avg": 30.4},
    {"Model": "Biomistral-7B", "medqa": 44.7, "medmcqa": 49.5, "mmlu": 53.1, "qa4mre": 68.6, "avg": 54.0},
    {"Model": "Meditron-70B", "medqa": 50.0, "medmcqa": 44.8, "mmlu": 79.6, "qa4mre": 51.4, "avg": 56.4},
    {"Model": "ClinicalCamel-70B", "medqa": 50.0, "medmcqa": 64.3, "mmlu": 83.7, "qa4mre": 68.6, "avg": 66.7},
    {"Model": "GPT-3.5-turbo w/ Ada", "medqa": 57.2, "medmcqa": 65.7, "mmlu": 83.7, "qa4mre": 62.9, "avg": 67.4},
    {"Model": "Almanac", "medqa": 48.0, "medmcqa": 69.5, "mmlu": 71.4, "qa4mre": 60.0, "avg": 62.2},
    {"Model": "Clinfo.ai", "medqa": 54.3, "medmcqa": 77.0, "mmlu": 81.3, "qa4mre": 67.7, "avg": 70.1},
    {"Model": "Clinfo.ai w/o PubMed API", "medqa": 49.3, "medmcqa": 68.6, "mmlu": 79.6, "qa4mre": 74.3, "avg": 67.9},
    {"Model": "GPT-3.5-turbo", "medqa": 50.0, "medmcqa": 71.9, "mmlu": 83.6, "qa4mre": 62.9, "avg": 67.1},
    {"Model": "DALK", "medqa": 57.9, "medmcqa": 75.2, "mmlu": 85.4, "qa4mre": 71.4, "avg": 72.6},
]


# --- 1. Types & Enums ---

class DatasetType(str, Enum):
    medqa = "medqa"
    medmcqa = "medmcqa"
    mmlu = "mmlu"
    qa4mre = "qa4mre"
    all = "all"


# --- 2. Processing Logic ---

def format_prompt(question: str, choices: List[str]) -> str:
    """Helper function to maintain a consistent prompt structure across all datasets."""
    letters = string.ascii_uppercase
    choices_formatted = "".join(
        [f"{letters[i]}) {choice}\n" for i, choice in enumerate(choices)]
    )
    
    return (
        f"You are a medical expert. Please answer the following multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_formatted}\n"
        f"Provide only the letter corresponding to the correct answer (e.g., A, B, C, or D)."
    )


def process_medqa_item(item) -> Tuple[str, str, str, List[str]]:
    question: str = item.get("question")
    if not question:
        raise ValueError(f"Question not found in item {item}")

    choices: list = item.get("choices")
    if not choices:
        raise ValueError(f"Choices not found in item {item}")
    choices = list(map(str.strip, choices))

    answer_text: str = item.get("answer", [None])[0]
    if not answer_text:
        raise ValueError(f"Answer not found in item {item}")

    answer_index: int = choices.index(answer_text.strip())
    expected_letter = string.ascii_uppercase[answer_index]
    prompt = format_prompt(question, choices)

    return prompt, expected_letter, question, choices


def process_medmcqa_item(item: Dict[str, Any]) -> Tuple[str, str, str, List[str]]:
    question: str = item.get("question")
    if not question:
        raise ValueError(f"Question not found in item {item}")

    opa = item.get("opa")
    if not opa:
        raise ValueError(f"Option A not found in item {item}")

    opb = item.get("opb")
    if not opb:
        raise ValueError(f"Option B not found in item {item}")

    opc = item.get("opc")
    if not opc:
        raise ValueError(f"Option C not found in item {item}")

    opd = item.get("opd")
    if not opd:
        raise ValueError(f"Option D not found in item {item}")

    choices = [opa, opb, opc, opd]

    correct_option_index = item.get("cop")
    if correct_option_index is None or not (0 <= int(correct_option_index) <= 3):
        raise ValueError(f"Correct Option not found in item {item}")

    expected_letter = string.ascii_uppercase[int(correct_option_index)]
    prompt = format_prompt(question, choices)

    return prompt, expected_letter, question, choices


def process_mmlu_item(item: Dict[str, Any]) -> Tuple[str, str, str, List[str]]:
    question: str = item.get("question")
    if not question:
        raise ValueError(f"Question not found in item {item}")

    choices: list = item.get("choices")
    if not choices:
        raise ValueError(f"Choices not found in item {item}")
    choices = list(map(str.strip, choices))

    answer_index = item.get("answer")
    if answer_index is None or not (0 <= int(answer_index) <= 3):
        raise ValueError(f"Answer not found in item {item}")

    expected_letter = string.ascii_uppercase[int(answer_index)]
    prompt = format_prompt(question, choices)

    return prompt, expected_letter, question, choices


def process_qa4mre_item(item: Dict[str, Any]) -> Tuple[str, str, str, List[str]]:
    context = item.get("document_str", "")
    base_question = item.get("question_str")
    
    if not base_question:
        raise ValueError(f"Question not found in item {item}")

    # NOTE: The retrieved document context (`context` / `document_str`) is deliberately omitted from 
    # the question prompt here. Empirical evaluations and trials showed that providing the large 
    # context actually degraded model performance on QA4MRE questions compared to raw isolation.
    question = f"{base_question}"

    answer_options = item.get("answer_options")
    if not answer_options:
        raise ValueError(f"Answer Options not found in {item}")

    correct_answer_id = item.get("correct_answer_id")
    if not correct_answer_id:
        raise ValueError(f"Correct Answer ID not found in {item}")
    
    correct_answer_id = str(correct_answer_id)

    choices = []
    expected_letter = None
    
    answer_ids = answer_options.get("answer_id", [])
    answer_strs = answer_options.get("answer_str", [])
    
    if len(answer_ids) != len(answer_strs) or not answer_strs:
        raise ValueError(f"Malformed answer_options arrays in {item}")

    for idx, (ans_id, text) in enumerate(zip(answer_ids, answer_strs)):
        if not text:
            raise ValueError(f"Text not found in one option. Item {item}")

        choices.append(text)

        if str(ans_id) == correct_answer_id:
            expected_letter = string.ascii_uppercase[idx]

    prompt = format_prompt(question, choices)

    return prompt, expected_letter, question, choices


PROCESSORS = {
    DatasetType.medqa: process_medqa_item,
    DatasetType.medmcqa: process_medmcqa_item,
    DatasetType.mmlu: process_mmlu_item,
    DatasetType.qa4mre: process_qa4mre_item,
}


# --- 3. Prompts & API Core Handlers ---

async def execute_llm_api_call(prompt: str) -> str:
    """Executes the async GET request to the local LLM RAG API."""
    params = {"query": prompt}
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("answer", "ERROR: Unexpected API Response Format")


async def execute_prompt(prompt: str, direct: bool = True) -> str:
    """
    Executes the prompt evaluation.
    - If direct=True: invokes the in-process Python pipeline directly.
    - If direct=False: sends an HTTP request to the local server.
    """
    if direct:
        from src.pipelines.rag_pipeline import BosnaRagPipeline
        from asyncer import asyncify

        pipeline = BosnaRagPipeline()
        res = await asyncify(pipeline.generate_enhanced_response)(prompt, experiment_id="benchmark")
        if not res or not isinstance(res, dict):
            raise ValueError("Invalid or empty response dict returned from local pipeline")
        return res.get("answer", "ERROR: Unexpected API Response Format")
    else:
        return await execute_llm_api_call(prompt)


async def call_api_with_retry(
    question_id: str, 
    prompt: str, 
    direct: bool = True,
    max_retries: int = 3, 
    max_transient_retries: int = 30, 
    base_delay: float = 1.0
) -> str:
    """
    Executes an API call with exponential backoff and localized error telemetry.
    Implements a capped backoff for transient network errors to prevent infinite hanging.
    """
    attempt = 0
    total_iterations = 0
    
    transient_errors = (
        "TimeoutError", 
        "ClientResponseError", 
        "ServerDisconnectedError", 
        "ClientConnectorError", 
        "ServerTimeoutError"
    )

    while True:
        try:
            return await execute_prompt(prompt, direct=direct)

        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            
            is_transient = error_name in transient_errors
            short_prompt = prompt[:50].replace('\n', ' ') + "..."
            
            logger.error(
                f"QuestionID: {question_id} | Attempt: {total_iterations + 1} | "
                f"Error: {error_name} - {error_msg} | Prompt: '{short_prompt}'"
            )
            
            # --- The Circuit Breaker ---
            if total_iterations >= max_transient_retries:
                logger.error(f"CIRCUIT BREAKER TRIPPED | QuestionID: {question_id} | Max transient retries ({max_transient_retries}) exhausted.")
                return f"ERROR: {error_name}_TIMEOUT"
                
            if not is_transient:
                attempt += 1
                
            # If max_retries is reached and the error is NOT transient, break and return
            if attempt >= max_retries and not is_transient:
                logger.error(f"FINAL ABORT | QuestionID: {question_id} | Giving up on prompt: '{short_prompt}'")
                return f"ERROR: {error_name}"
            
            # Exponential backoff calculation
            delay = base_delay * (2 ** min(total_iterations, 6))
            await asyncio.sleep(delay)
            total_iterations += 1


def get_deterministic_id(item: Dict[str, Any], prompt: str) -> str:
    """Returns the item ID if it exists, otherwise hashes the prompt."""
    if "id" in item:
        return str(item["id"])
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def load_completed_ids(jsonl_path: Path) -> Set[str]:
    """Loads previously processed evaluation IDs to enable resuming."""
    completed = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed.add(record.get("id"))
                    except json.JSONDecodeError:
                        continue
    return completed


# --- 4. Concurrency & Execution Core ---

async def evaluate_single_attempt(
    base_id: str,
    dataset_name: str,
    attempt_index: int,
    prompt: str,
    question: str,
    choices: List[str],
    expected: Optional[str],
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    out_path: Path,
    progress_callback: Optional[Any] = None,
    direct: bool = True
):
    """Handles a single API call, writes the expanded result, and triggers progress callback."""
    async with semaphore:
        received_answer = await call_api_with_retry(base_id, prompt, direct=direct)

    is_correct = (received_answer.strip().upper() == expected) if expected else False
    attempt_id = f"{dataset_name}_{base_id}_{attempt_index}"

    record = {
        "id": attempt_id,
        "base_id": base_id,
        "dataset": dataset_name,
        "attempt": attempt_index,
        "question": question,
        "choices": choices,
        "prompt": prompt,
        "expected_answer": expected,
        "received_answer": received_answer,
        "is_correct": is_correct
    }

    async with file_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    if progress_callback:
        try:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback()
            else:
                progress_callback()
        except Exception:
            pass


async def process_dataset(
    dataset_name: DatasetType,
    data: List[Dict[str, Any]],
    completed_ids: Set[str],
    out_path: Path,
    n_calls: int,
    concurrency_limit: int,
    progress_callback: Optional[Any] = None,
    direct: bool = True
):
    """Orchestrates the evaluation of a single dataset using concurrency controls."""
    processor = PROCESSORS[dataset_name]
    semaphore = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    tasks = []

    for item in data:
        prompt, expected, question_text, choices_list = processor(item)
        base_id = get_deterministic_id(item, prompt)
        
        for i in range(n_calls):
            attempt_id = f"{dataset_name.value}_{base_id}_{i}"
            
            if attempt_id in completed_ids:
                continue
                
            tasks.append(
                evaluate_single_attempt(
                    base_id=base_id,
                    dataset_name=dataset_name.value,
                    attempt_index=i,
                    prompt=prompt,
                    question=question_text,
                    choices=choices_list,
                    expected=expected,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    out_path=out_path,
                    progress_callback=progress_callback,
                    direct=direct
                )
            )

    if tasks:
        await asyncio.gather(*tasks)


# --- 5. Metrics & Visualization Engine ---

def generate_deterministic_plot(
    input_file: Path,
    attempt_id: int,
    dataset: DatasetType,
    output_plot: Path
) -> Dict[str, Any]:
    """Generates the single-trial accuracy plot and returns numerical summaries."""
    if not input_file.exists():
        raise FileNotFoundError(f"Target Excel file '{input_file}' not found.")

    data = pd.read_excel(input_file)
    data = data.dropna(subset=['expected_answer', 'received_answer'])

    # 1. Isolate the specific trial
    df_deterministic = data[data['attempt'] == attempt_id].copy()
    if df_deterministic.empty:
         raise ValueError(f"No data found in consolidated Excel for attempt {attempt_id}.")

    # 2. Compute empirical hit rate (converted to percentage)
    metrics = df_deterministic.groupby('dataset')['is_correct'].mean().reset_index()
    metrics['empirical_hit_rate'] = metrics['is_correct'] * 100
    our_avg = metrics['empirical_hit_rate'].mean()

    # 3. Construct "Our System" row
    our_system_dict = {"Model": "Our System"}
    for _, row in metrics.iterrows():
        our_system_dict[row['dataset'].lower()] = row['empirical_hit_rate']
    our_system_dict["avg"] = our_avg

    # 4. Merge with established benchmarks    
    df_bench = pd.DataFrame(BENCHMARK_DATA)
    df_our = pd.DataFrame([our_system_dict])
    df_combined = pd.concat([df_bench, df_our], ignore_index=True)

    # 5. Plotting Logic
    sns.set_theme(style="whitegrid")

    results_summary = {
        "our_system": our_system_dict,
        "metrics_per_dataset": metrics.to_dict(orient="records"),
        "plot_path": str(output_plot)
    }
        
    if dataset != DatasetType.all:
        target_col = dataset.value.lower()
        if target_col not in df_combined.columns:
             raise ValueError(f"Selected dataset '{dataset.value}' is not present in benchmark data.")

        df_plot = df_combined[['Model', target_col]].dropna().copy()

        # Calculate standard competition rank
        df_plot['Rank'] = df_plot[target_col].rank(method='min', ascending=False).astype(int)
        df_plot = df_plot.sort_values(by=target_col, ascending=False).reset_index(drop=True)

        our_rank = df_plot.loc[df_plot['Model'] == "Our System", 'Rank'].values[0]
        total_models = len(df_plot)

        # Explicit color mapping
        viridis_colors = sns.color_palette("viridis", total_models)
        color_dict = {
            model: 'crimson' if model == "Our System" else viridis_colors[i] 
            for i, model in enumerate(df_plot['Model'])
        }

        plt.figure(figsize=(10, 8))
        ax = sns.barplot(data=df_plot, x=target_col, y="Model", hue="Model", palette=color_dict, legend=False)

        ax.set_title(f"Deterministic Hit-Rate: {dataset.value.upper()} (Attempt {attempt_id})\nRank: {our_rank} of {total_models}", pad=20, fontsize=14, fontweight="bold")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("")

        # Add bar labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

        # Highlight text
        for tick_label in ax.get_yticklabels():
            if "Our System" in tick_label.get_text():
                tick_label.set_color('crimson')
                tick_label.set_fontweight('bold')
                
        results_summary["rank"] = int(our_rank)
        results_summary["total_models"] = total_models

    else:
        # Melt the dataframe for facet processing
        df_melted = df_combined.melt(id_vars="Model", var_name="Dataset", value_name="Accuracy").dropna()
        df_melted['Dataset'] = df_melted['Dataset'].str.upper()

        datasets = df_melted['Dataset'].unique()
        n_cols = 3
        n_rows = math.ceil(len(datasets) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        # Handle standard numpy flattening regardless of dimension
        axes = np.array(axes).flatten()

        for idx, ds_name in enumerate(datasets):
            ax = axes[idx]
            df_ds = df_melted[df_melted['Dataset'] == ds_name].copy()

            # Calculate standard competition rank
            df_ds['Rank'] = df_ds['Accuracy'].rank(method='min', ascending=False).astype(int)
            df_ds = df_ds.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
            
            our_rank = df_ds.loc[df_ds['Model'] == "Our System", 'Rank'].values[0]
            total_models = len(df_ds)
            
            # Explicit color mapping
            viridis_colors = sns.color_palette("viridis", total_models)
            color_dict = {
                model: 'crimson' if model == "Our System" else viridis_colors[i] 
                for i, model in enumerate(df_ds['Model'])
            }

            sns.barplot(data=df_ds, x='Accuracy', y='Model', hue='Model', palette=color_dict, legend=False, ax=ax)

            ax.set_title(f"{ds_name} | Rank: {our_rank} of {total_models}", pad=10, fontweight="bold")
            ax.set_xlabel("Accuracy (%)")
            ax.set_ylabel("")

            # Add bar labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3, size=9)

            # Highlight text
            for tick_label in ax.get_yticklabels():
                if "Our System" in tick_label.get_text():
                    tick_label.set_color('crimson')
                    tick_label.set_fontweight('bold')

        # Remove empty subplots if the grid is larger than the number of datasets
        for idx in range(len(datasets), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        fig.suptitle(f"Global Deterministic Benchmark Comparison (Attempt {attempt_id})", y=1.03, fontsize=16, fontweight="bold")

    # Make parent directory if it does not exist
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    return results_summary


def evaluate_distributional_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Expected Accuracy, Entropy, Consistency, and Brier Score per question & globally."""
    epsilon = 1e-9
    results = []
    
    dataset_cardinalities = {}
    for ds in df['dataset'].unique():
        ds_data = df[df['dataset'] == ds]
        all_options = set(ds_data['expected_answer'].dropna().unique()) | set(ds_data['received_answer'].dropna().unique())
        dataset_cardinalities[ds] = len(all_options)
    
    grouped = df.groupby(['dataset', 'base_id'])
    
    for (dataset, base_id), group in grouped:
        expected_ans = group['expected_answer'].iloc[0]
        pmf = group['received_answer'].value_counts(normalize=True)
        
        acc_exp = pmf.get(expected_ans, 0.0)
        probabilities = pmf.values
        entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
        
        num_choices = dataset_cardinalities[dataset]
        max_entropy = np.log2(num_choices) if num_choices > 1 else 1.0
        normalized_consistency = 1.0 - (entropy / max_entropy)
        
        brier_score = 0.0
        active_classes = set(pmf.index).union({expected_ans})
        
        for c in active_classes:
            p_c = pmf.get(c, 0.0)
            delta = 1.0 if c == expected_ans else 0.0
            brier_score += (p_c - delta) ** 2
            
        results.append({
            'dataset': dataset,
            'base_id': base_id,
            'expected_accuracy': acc_exp,
            'entropy': entropy,
            'consistency_score': normalized_consistency,
            'brier_score': brier_score
        })
        
    dist_df = pd.DataFrame(results)
    
    global_metrics = dist_df.groupby('dataset').agg(
        mean_expected_accuracy=pd.NamedAgg(column='expected_accuracy', aggfunc='mean'),
        mean_entropy=pd.NamedAgg(column='entropy', aggfunc='mean'),
        mean_consistency=pd.NamedAgg(column='consistency_score', aggfunc='mean'),
        mean_brier_score=pd.NamedAgg(column='brier_score', aggfunc='mean')
    ).reset_index()
    
    return dist_df, global_metrics


def generate_distributional_plot(
    input_file: Path,
    dataset: DatasetType,
    output_plot: Path
) -> Dict[str, Any]:
    """Generates the multi-attempt consistency distributions and returns metrics."""
    if not input_file.exists():
        raise FileNotFoundError(f"Target Excel file '{input_file}' not found.")

    data = pd.read_excel(input_file)
    data = data.dropna(subset=['expected_answer', 'received_answer'])
    
    dist_df, global_metrics = evaluate_distributional_metrics(data)
    if dist_df.empty:
         raise ValueError("No distributional data could be computed.")
         
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.array(axes).flatten()
    
    metrics_info = [
        ("expected_accuracy", "Expected Accuracy (%)", True),
        ("consistency_score", "Consistency Score (%)", True),
        ("entropy", "Shannon Entropy (Bits)", False),
        ("brier_score", "Brier Score (Lower is Better)", False)
    ]

    if dataset != DatasetType.all:
        target_col = dataset.value.lower()
        df_plot = dist_df[dist_df['dataset'].str.lower() == target_col].copy()
        
        if df_plot.empty:
            raise ValueError(f"No data found for dataset '{dataset.value}'.")
            
        for idx, (col, title, is_pct) in enumerate(metrics_info):
            ax = axes[idx]
            plot_val = df_plot[col] * 100 if is_pct else df_plot[col]
            sns.histplot(plot_val, kde=True, ax=ax, color='crimson', bins=15, alpha=0.6)
            ax.set_title(f"Density of {title}\n({dataset.value.upper()})", pad=15, fontweight="bold")
            ax.set_xlabel(title)
            ax.set_ylabel("Frequency of Questions")
            
        fig.suptitle(f"Micro-Distributional Analysis: {dataset.value.upper()}", y=1.03, fontsize=16, fontweight="bold")

    else:
        df_plot = dist_df.copy()
        
        for idx, (col, title, is_pct) in enumerate(metrics_info):
            ax = axes[idx]
            df_plot['plot_val'] = df_plot[col] * 100 if is_pct else df_plot[col]
            
            is_ascending = col in ['entropy', 'brier_score']
            order = df_plot.groupby('dataset')['plot_val'].mean().sort_values(ascending=is_ascending).index
            
            total_datasets = len(order)
            palette = sns.color_palette("viridis", total_datasets)
            
            sns.barplot(
                data=df_plot, x='plot_val', y='dataset', 
                order=order, palette=palette, ax=ax, capsize=.1, errorbar=('ci', 95)
            )
            
            ax.set_title(f"Mean {title}\n(with 95% CI)", pad=15, fontweight="bold")
            ax.set_xlabel(title)
            ax.set_ylabel("")
            
            if ax.containers:
                ax.bar_label(ax.containers[0], fmt='%.1f' if is_pct else '%.3f', padding=5, size=10)

        fig.suptitle("Global Distributional Metrics Comparison (PMF across N-Trials)", y=1.03, fontsize=16, fontweight="bold")

    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "global_metrics": global_metrics.to_dict(orient="records"),
        "plot_path": str(output_plot)
    }


def analyze_telemetry_errors(
    log_path: Path,
    output_plot: Path
) -> List[Dict[str, Any]]:
    """Parses transaction logs and plots peak retry errors per QuestionID."""
    if not log_path.exists():
        raise FileNotFoundError(f"The log file {log_path} does not exist.")

    log_pattern = re.compile(r"QuestionID:\s*([^|]+)\s*\|\s*Attempt:\s*(\d+)\s*\|\s*Error:\s*([^|-]+)")
    
    parsed_data = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                parsed_data.append({
                    "question_id": match.group(1).strip(),
                    "attempt": int(match.group(2).strip()),
                    "error_class": match.group(3).strip()
                })
                
    if not parsed_data:
        return []
        
    df = pd.DataFrame(parsed_data)
    max_attempts_df = df.groupby(["question_id", "error_class"])["attempt"].max().reset_index()
    critical_failures = max_attempts_df.sort_values(by="attempt", ascending=False).head(20)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(
        data=critical_failures, 
        x="attempt", 
        y="question_id", 
        hue="error_class", 
        dodge=False, 
        palette="magma"
    )
    
    ax.set_title("Telemetry Analysis: Top 20 Transactional Bottlenecks", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Maximum Recorded Attempts (Exponential Backoff)")
    ax.set_ylabel("Question Identifier (Question ID)")
    
    for container in ax.containers:
        ax.bar_label(container, padding=5, size=10)
        
    plt.legend(title="Exception Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    
    return critical_failures.to_dict(orient="records")
