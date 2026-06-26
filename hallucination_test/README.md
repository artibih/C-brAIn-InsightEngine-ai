# Hypothesis Evaluation experiment

This repository contains a high-performance, asynchronous CLI pipeline designed to evaluate generative AI system outputs against a curated dataset of composite hypotheses.
The system strictly tracks the convergence of the `hallucination_detector` agent across iterative refinement cycles.

## 1. Experimental Design

The evaluation was based on a ground truth dataset derived from 100 source papers.
From these, 2,000 discrete claims were extracted (20 per paper) and classified against the source material, resulting in 1,470 ENTAILED and 530 NEUTRAL claims.
To introduce a robust adversarial component, 350 CONTRADICTORY claims were synthetically engineered from the ENTAILED baseline. 

From this curated pool, a test set of 1,000 composite hypotheses was constructed, with each hypothesis functioning as a mapped output of 5 underlying claims.
To evaluate the system's boundary detection, the test set was strictly stratified into three uniform cohorts:

* **33% Control cohort** (derived exclusively from ENTAILED claims).
* **33% Treatment cohort A** (injected with a single NEUTRAL claim).
* **33% Treatment cohort B** (injected with a single CONTRADICTORY claim).

## 2. Input Data Schema

The pipeline requires a source file named `Hypothesis.xlsx` by default.
It must contain the following exact column structure:

| Column Name | Description |
| :--- | :--- |
| **Hypothesis ID** | Unique identifier of the hypothesis. |
| **Hypothesis** | The text payload to test the system against. |
| **Global Classification** | Describes the composite injection state: `ENTAILMENT-based`, `NEUTRAL-based`, or `CONTRADICTION-based`. |
| **Claim ID** | Array/List of unique identifiers for the underlying claims used to construct the hypothesis. |
| **Claim** | Array/List of exact text claims extracted from the sources. |
| **Source paper title** | Array/List of the source document titles. |
| **Source (PDF name)** | Array/List of the source document filenames. |
| **Classification** | Array/List of categorical mappings defining the truth value of each individual claim. |
| **Why** | Array/List of rationales explaining the assigned classifications. |

### Classification Definitions:

* **ENTAILMENT:** A TRUE assertion that can be logically concluded from the corresponding source.
* **CONTRADICTION:** A FALSE assertion that cannot be logically concluded from the corresponding source.
* **NEUTRAL:** An INDETERMINATE assertion taking into account only the corresponding source.

## 3. System Requirements & Setup

The execution environment is optimized for low-resource constraints utilizing vectorized memory mapping via Polars and `calamine`, enabling experiments of bigger size. 

Dependency management is handled via `uv` to ensure deterministic builds.
All required dependencies are already declared in the `pyproject.toml` file.
To configure the isolated virtual environment and install the exact package graph, execute:

```bash
uv sync
```

## 4. Execution Methodology

The system is decoupled into three idempotent phases accessible via the `typer` CLI.
You can view the schema definitions and parameters directly in the terminal by running:

```bash
uv run python hallucination_evaluator_cli.py [COMMAND] --help
```

### Phase 1: Evaluation
Executes the asynchronous test stream.
The system parses Server-Sent Events (SSE) line-by-line, buffering results iteratively to a temporary NDJSON file (`temp_evaluation_results.jsonl`) to ensure O(1) memory complexity and robust fault tolerance.

```bash
uv run python hallucination_evaluator_cli.py evaluate
```

> **Note:** If the execution is interrupted, re-running this command will automatically hydrate the state from the NDJSON file and resume precisely where it halted via a vectorized anti-join.

### Phase 2: Consolidation

Converts the intermediate NDJSON state into a final, structured Excel artifact containing the iterative metrics mapped against the original schema.

```bash
uv run python hallucination_evaluator_cli.py consolidate
```

### Phase 3: Analytics Generation

Generates a statistical distribution plot tracking the convergence of hallucinations across the 3 sequential agent calls, grouped by `Global Classification`.

```bash
uv run python hallucination_evaluator_cli.py plot
```

> **Outputs:** `hallucination_analysis.png`
