import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timezone
import aiohttp
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typer

app = typer.Typer(help="MLOps CLI for Hypothesis Evaluation and Result Analysis")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def process_hypothesis(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, row_data: dict, max_retries: int = 3) -> tuple[dict | None, dict | None]:
    url = 'http://localhost:8000/api/v1/hypothesis/test/stream'
    payload = {"hypothesis": row_data['Hypothesis'], "async_execution": True}
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    result_record = row_data.copy()
    metrics = ["total_claims", "entailed", "contradicted", "neutral", "hallucination_risk_score"]
    for i in range(1, 4):
        for metric in metrics:
            result_record[f"call_{i}_{metric}"] = None
    last_agent = "unknown"
    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    call_counter = 1
                    async for line in response.content:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data:'):
                            json_str = decoded_line[5:].strip()
                            if not json_str: continue
                            try:
                                data = json.loads(json_str)
                                last_agent = data.get('agent', last_agent) 
                                if last_agent == 'hallucination_detector' and 'summary' in data:
                                    summary = data['summary']
                                    prefix = f"call_{call_counter}"
                                    if call_counter <= 3:
                                        for metric in metrics:
                                            result_record[f"{prefix}_{metric}"] = summary.get(metric)
                                    call_counter += 1
                            except json.JSONDecodeError:
                                logging.warning(f"Malformed JSON payload for ID {row_data['Hypothesis ID']}")
                return result_record, None
            except Exception as e:
                logging.warning(f"Attempt {attempt}/{max_retries} failed for ID {row_data['Hypothesis ID']}: {str(e)}")
                if attempt == max_retries:
                    error_record = {
                        "hypothesis_id": row_data['Hypothesis ID'],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "last_agent": last_agent,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    return None, error_record
                await asyncio.sleep(2 ** attempt)

async def _evaluate_async(input_file: str, temp_ndjson: str, error_log: str, max_retries: int):
    logging.info(f"Mapping source dataset from {input_file}")
    df = pl.read_excel(input_file, engine="calamine")
    processed_ids = set()
    if os.path.exists(temp_ndjson):
        with open(temp_ndjson, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(str(record.get('Hypothesis ID')))
                    except json.JSONDecodeError:
                        continue
    initial_count = len(df)
    df = df.filter(~pl.col('Hypothesis ID').cast(pl.Utf8).is_in(processed_ids))
    remaining_count = len(df)
    logging.info(f"State Hydration: Skipped {initial_count - remaining_count} processed records. {remaining_count} pending.")
    if remaining_count == 0:
        logging.info("Evaluation complete. No pending records.")
        return
    rows = df.to_dicts()
    semaphore = asyncio.Semaphore(15) 
    tasks = []
    timeout = aiohttp.ClientTimeout(total=None) 
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for row in rows:
            tasks.append(process_hypothesis(session, semaphore, row, max_retries))
        logging.info(f"Dispatching {len(tasks)} asynchronous tasks...")
        for future in asyncio.as_completed(tasks):
            success_result, error_result = await future
            if success_result:
                pl.DataFrame([success_result]).write_ndjson(open(temp_ndjson, mode='a', encoding='utf-8'))
                logging.info(f"Task complete for ID: {success_result.get('Hypothesis ID')}")
            if error_result:
                with open(error_log, mode='a', encoding='utf-8') as ef:
                    ef.write(json.dumps(error_result) + "\n")
                logging.error(f"Task permanently failed for ID: {error_result.get('hypothesis_id')}")

@app.command()
def evaluate(
    input_file: str = "Hypothesis.xlsx", 
    temp_ndjson: str = "temp_evaluation_results.jsonl", 
    error_log: str = "evaluation_errors.jsonl",
    max_retries: int = 3
):
    """
    Executes the asynchronous stream evaluation against the target API.
    
    EXPECTED INPUT SCHEMA (Excel):
    
    \b
    - Hypothesis ID: Unique identifier of the hypothesis.
    - Hypothesis: The text payload to test the system against.
    - Global Classification: ENTAILMENT-based, NEUTRAL-based, or CONTRADICTION-based.
    - Claim ID: List of unique identifiers for the underlying claims.
    - Claim: List of extracted source claims.
    - Source paper title: List of source document titles.
    - Source (PDF name): List of source filenames.
    - Classification: List of individual claim classifications (ENTAILMENT / NEUTRAL / CONTRADICTION).
    - Why: List of classification rationales.
    
    CLASSIFICATION DEFINITIONS:
    - ENTAILMENT: A TRUE assertion logically concluded from the source.
    - CONTRADICTION: A FALSE assertion logically disjoint from the source.
    - NEUTRAL: An INDETERMINATE assertion taking into account only the source.
    """
    asyncio.run(_evaluate_async(input_file, temp_ndjson, error_log, max_retries))

@app.command()
def consolidate(
    temp_ndjson: str = "temp_evaluation_results.jsonl", 
    output_file: str = "Evaluation_Results.xlsx"
):
    """Compiles the partial NDJSON evaluation state into the final Excel artifact."""
    if not os.path.exists(temp_ndjson):
        logging.error(f"Source file {temp_ndjson} not found.")
        raise typer.Exit(code=1)
        
    logging.info(f"Consolidating {temp_ndjson} into {output_file}")
    df = pl.read_ndjson(temp_ndjson)
    df.write_excel(output_file)
    logging.info("Consolidation complete.")

@app.command()
def plot(
    input_excel: str = "Evaluation_Results.xlsx", 
    output_png: str = "hallucination_analysis.png"
):
    """Calculates metrics and generates the hallucination distribution plot."""
    if not os.path.exists(input_excel):
        logging.error(f"Source file {input_excel} not found. Run 'consolidate' first.")
        raise typer.Exit(code=1)
    logging.info(f"Loading analytics data from {input_excel}")
    df = pd.read_excel(input_excel)
    cols_to_fix = ['call_1_contradicted', 'call_1_neutral', 'call_1_total_claims', 'call_2_contradicted', 'call_2_neutral', 'call_2_total_claims', 'call_3_contradicted', 'call_3_neutral', 'call_3_total_claims']
    for col in cols_to_fix: df[col] = pd.to_numeric(df[col], errors='coerce')
    for i in range(1, 4): df[f'call_{i}_hallucination_detected_pct'] = ((df[f'call_{i}_contradicted'] + df[f'call_{i}_neutral']) / df[f'call_{i}_total_claims']) * 100
    plot_df = df.melt(id_vars=['Global Classification'], value_vars=['call_1_hallucination_detected_pct', 'call_2_hallucination_detected_pct', 'call_3_hallucination_detected_pct'], var_name='Call', value_name='Hallucinations (% over total claims)')
    plot_df['Call'] = plot_df['Call'].replace({'call_1_hallucination_detected_pct': 'Call 1', 'call_2_hallucination_detected_pct': 'Call 2', 'call_3_hallucination_detected_pct': 'Call 3'})
    logging.info("Rendering plot distributions...")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=plot_df, x='Global Classification', y='Hallucinations (% over total claims)', hue='Call')
    plt.title('Distribution of Hallucinations (% over total claims) by Global Classification and Call')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    logging.info(f"Plot saved successfully to {output_png}")

if __name__ == "__main__":
    app()