# ADQA Benchmarking CLI

## 1. Overview and Purpose
This CLI is a high-performance orchestration tool designed to evaluate the performance of the Retrieval Augmented LLM model with documented retrieval datasets against
**ADQA benchmark**, an Alzheimer's Disease specialized multiple-choice question-answering datasets (MedQA, MedMCQA, MMLU, QA4MRE).

Beyond standard deterministic benchmarking (checking if an answer is right or wrong), this system is engineered to evaluate the **probabilistic topology** of the model's generations.
By querying the model $N$ times, the CLI constructs an empirical **Probability Mass Function (PMF)** for every individual question.
This allows researchers to quantify the model's accuracy, uncertainty, calibration, and stability.

## 2. CLI and Datasets location
ADQA benchmark CLI is located inside the main repo in the `.adqa_benchmark` folder.
ADQA datasets are located in the `./adqa_benchmark/data_folder` as JSON files named after the corresponding dataset: MedQA, MedMCQA, MMLU, QA4MRE.

## 3. Environment Setup (using `uv`)
To ensure strict dependency resolution and high-performance virtual environment management, this project uses `uv`.

```bash
# 1. Install uv (if not already installed)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 2. From main repo folder, move to adqa_benchmark folder
cd adqa_benchmark

# 3. Sync uv project
uv sync

# 4. Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\\Scripts\\activate
```

# 3. CLI Commands and Execution Syntaxes

> [!IMPORTANT]  
> From now on, we assume that you're inside the `./adqa_benchmark` folder with the `(adqa_benchmark)` virtual environment already activated.
> We also assume you're running the system locally.

The CLI comes with internal help menu you can consult:

```bash
python adqa_benchmark.py --help
```

You will see that the following commands are available:

- `run`: Evaluates the LLM asynchronously with resilient retry and resumption capabilities.
- `info`: Displays the total number of questions available in the specified dataset(s).
- `export`: Consolidates the temporary JSONL results into a single Excel file.
- `deterministic`: Evaluates the deterministic accuracy (single trial) and benchmarks it visually against SOTA models.
- `distributional`: Evaluates the N-sample PMF metrics and benchmarks the system's uncertainty and probabilistic consistency visually.

The expected output when running `info` command is:

```bash
# Command
python adqa_benchmark_cli.py info all --source ./data_folder

# Command output
Dataset Information

┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Dataset ┃ Total Questions ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ medqa   │             152 │
│ medmcqa │             210 │
│ mmlu    │              49 │
│ qa4mre  │              35 │
└─────────┴─────────────────┘
```

## 4. Experiment workflow
The experiment workflow is divided into logical phases: Evaluation, Consolidation, and Analytics.

### Phase 1: Asynchronous Evaluation
Streams prompts to the target API asynchronously, implementing a retry mechanism with local error logging (`benchmark_errors.log`) to ensure fault tolerance.

```bash
python adqa_benchmark.py run [OPTIONS] DATASET:{medqa|medmcqa|mmlu|qa4mre|all}
```

For example:

```bash
python adqa_benchmark.py run all -s ./data_folder
```

- **Expected outcome:** JSONL file (`results_temp.jsonl`) with unformatted experiment results.

> [!TIP]  
> If experiment fails during the execution, you can run your experiment again with the same command and it will automatically detect
> which questions are missing for evaluation, in order to resume the experiment.

> [!TIP]  
> Run command-specific help for more info:
>
> ```bash
> python adqa_benchmark.py run --help
> ```

### Phase 2: Consolidation
Consolidates the temporary JSONL results into a single Excel file. JSONL results are deleted after this step.

```bash
python adqa_benchmark_cli.py export [OPTIONS]
```

- **Expected outcome:** Excel file (`results_consolidated.xlsx`) with analysis-ready, consolidated results of the experiment.

> [!TIP]  
> Run command-specific help for more info:
>
> ```bash
> python adqa_benchmark.py export --help
> ```

### Phase 3: Analytics
Evaluates the deterministic accuracy (single trial) and benchmarks it visually against SOTA models.

```bash
python adqa_benchmark_cli.py deterministic [OPTIONS]
```

- **Expected outcome:** PNG file (`deterministic_benchmark.png`) with the generated seaborn plot analyzing the ADQA benchmark experiment result against SOTA models.

> [!TIP]  
> Run command-specific help for more info:
>
> ```bash
> python adqa_benchmark.py deterministic --help
> ```
