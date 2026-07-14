# SchemaRAG Runbook and Extension Summary

## 1. How to run the codebase and evaluate it

1. Set up a Python virtual environment in the repository root and install dependencies from `requirements.txt`.
2. Put the data and weights in the paths the scripts already expect, or be ready to pass them as CLI arguments.
3. Train or download SchemaLinker.
4. Train SAR and save the two checkpoints and embedding caches.
5. Run SAR inference to create `retrieval_results.json`.
6. Run `main.py` to generate SQL predictions.
7. Run the evaluation scripts to get accuracy and VES.

### Step 1. Environment setup

Run this from the repository root:

```bash
cd /Users/suhasreddy/Downloads/SchemaRAG-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2. Where to put data and weights

The repo uses these default locations in code:

- `./SAR/models/best_schema_aware_model.pth`
- `./SAR/models/best_contrastive_model.pth`
- `./plm/embeddingmodel`
- `./retrieval_results.json`
- `./data_with_sk.json`
- `./log/predicted.sql`
- `./training_plots/`
- `./logs/`

For the BIRD evaluator, `run_evaluation.sh` expects:

- `./data/bird/database/`
- `./data/bird/dev.json`
- `./log/bird/`

If you want to keep the repo unchanged, place your files in those exact paths. Otherwise, edit the paths in the scripts before running.

### Step 3. SchemaLinker download or training

If the pretrained SchemaLinker download is available, use the README command:

```python
from modelscope import snapshot_download
model_dir = snapshot_download('TonyTANG11/SchemaLinker')
```

That downloads the model into the ModelScope cache and returns the local directory path in `model_dir`.

If you need to train SchemaLinker yourself, run the stages in this order:

1. CoT fine-tuning with [SchemaLinker_train/train_SchemaLinker_CoT_finetune.py](SchemaLinker_train/train_SchemaLinker_CoT_finetune.py).
2. MTL fine-tuning with [SchemaLinker_train/train_SchemaLinker_MTL_finetune.py](SchemaLinker_train/train_SchemaLinker_MTL_finetune.py).
3. GRPO training with [SchemaLinker_train/train_SchemaLinker_GRPO.py](SchemaLinker_train/train_SchemaLinker_GRPO.py) or the PEFT variant.

Example command pattern for CoT fine-tuning:

```bash
python SchemaLinker_train/train_SchemaLinker_CoT_finetune.py
```

Before running, edit these variables in the script:

- `local_model_path = "/path/to/your/model"`
- `local_dataset_path_train = "/path/to/your/data/train_data.json"`
- `output_dir = "./qwen_finetune_CoT"`

For MTL fine-tuning, update:

- `local_model_path = "/path/to/your/qwen_finetune_CoT"`
- `local_dataset_path_train = "/path/to/your/data/train.json"`
- `output_dir = "./qwen_finetune_mtl"`

For GRPO, update:

- `model_name = "/path/to/your/qwen_finetune_mtl"`
- `local_dataset_path_train = "/path/to/your/training_data.json"`
- `output_dir = "./SchemaLinker"`

The inference script [use_SchemaLinker.py](use_SchemaLinker.py) expects the final model folder at `./SchemaLinker` unless you change `model_name`.

### Step 4. Train SAR locally

Train SAR from [SAR_train/train_SAR.py](SAR_train/train_SAR.py). The script creates and saves:

- Stage 1 checkpoint: `./SAR/models/best_schema_aware_model.pth`
- Stage 2 checkpoint: `./SAR/models/best_contrastive_model.pth`
- Stage 1 embeddings: `./embeddings/stage1_embeddings.pt`
- Stage 2 embeddings: `./embeddings/stage2_embeddings.pt`
- Training plots: `./training_plots/`

The training script defaults to these inputs if you do not override them:

- `./mini_SA_200.json`
- `./mini_CL_200.json`
- `./plm/embeddingmodl`  

Note: the default FlagEmbedding path in the training config is misspelled in the code as `embeddingmodl`, so you will usually want to pass a real model directory with `--flag_model_path /path/to/your/flagembedding-model`.

If those files do not exist in your workspace, you must provide your own dataset paths and a real FlagEmbedding model directory.

Example command:

```bash
python SAR_train/train_SAR.py \
  --stage1_data /path/to/your/stage1_data.json \
  --stage2_data /path/to/your/stage2_data.json \
  --stage1_embeddings ./embeddings/stage1_embeddings.pt \
  --stage2_embeddings ./embeddings/stage2_embeddings.pt \
  --flag_model_path /path/to/your/flagembedding-model \
  --stage1_model_path ./SAR/models/best_schema_aware_model.pth \
  --stage2_model_path ./SAR/models/best_contrastive_model.pth \
  --output_dir ./training_plots \
  --log_dir ./logs
```

The expected data shapes are:

- Stage 1: each record needs `question`, `query`, and `schema`.
- Stage 2: each record needs `question`, `query`, `schema`, and `similar`, where `similar` is a list of similar examples with their own `question`, `query`, and optional `schema`.

### Step 5. Generate retrieval examples with SAR

Run [SAR_use.py](SAR_use.py) after the SAR checkpoints exist.

The script expects these default paths:

- `stage1_model_path = './SAR/models/best_schema_aware_model.pth'`
- `stage2_model_path = './SAR/models/best_contrastive_model.pth'`
- `flag_model_path = './plm/embeddingmodel'`
- `supervised_data_path = './RAG_Spider_formatdata.json'`
- `test_file_path = './dev_spider.json'`
- `output_file_path = './retrieval_results.json'`

The file currently calls `example_usage()` in the `__main__` block, so if you want to use your own paths you should either edit the bottom of the file to call `main()` or run the module after adjusting `example_usage()`.

The output file must be named `retrieval_results.json` because [main.py](main.py) loads that file directly.

### Step 6. Run SQL generation

[main.py](main.py) currently uses these hardcoded inputs:

- `DATASET = "./data_with_sk.json"`
- `SIMILAR_FILE = "retrieval_results.json"`
- `OUTPUT_FILE = "log/predicted.sql"`

So before running, make sure:

- `./data_with_sk.json` exists.
- `./retrieval_results.json` exists.
- The `./log/` directory exists or can be created.

Then run:

```bash
python main.py
```

This produces SQL lines in `./log/predicted.sql`.

### Step 7. Run evaluation and get metrics

For BIRD, the repository provides [run_evaluation.sh](run_evaluation.sh).

Run:

```bash
bash run_evaluation.sh
```

That script calls both:

- [src/evaluation.py](src/evaluation.py) for execution accuracy
- [src/evaluation_ves.py](src/evaluation_ves.py) for VES

Both scripts expect the BIRD directory structure under `./data/bird/` and prediction files under `./log/bird/`.

If you are not using BIRD, run the evaluation scripts directly and pass your own paths:

```bash
python src/evaluation.py \
  --predicted_sql_path /path/to/predictions_dir \
  --ground_truth_path /path/to/ground_truth_dir \
  --data_mode dev \
  --db_root_path /path/to/database_root \
  --diff_json_path /path/to/difficulty.json \
  --num_cpus 16 \
  --meta_time_out 30.0
```

```bash
python src/evaluation_ves.py \
  --predicted_sql_path /path/to/predictions_dir \
  --ground_truth_path /path/to/ground_truth_dir \
  --data_mode dev \
  --db_root_path /path/to/database_root \
  --diff_json_path /path/to/difficulty.json \
  --num_cpus 16 \
  --meta_time_out 30.0
```

The metrics printed are:

- simple / moderate / challenging / total accuracy in `src/evaluation.py`
- simple / moderate / challenging / total VES in `src/evaluation_ves.py`

## 2. What the two added extensions do

### UARB: Uncertainty-Aware Retrieval Budgeting

UARB changes the retrieval stage in `SAR_use.py` from a fixed-k setup to a dynamic budget. It computes a Retrieval Uncertainty Score from the top candidate similarity distribution, then adjusts the number of retrieved examples based on how confident the retriever is.

Why it helps:
- High-confidence queries retrieve fewer examples and avoid noise.
- Ambiguous queries retrieve more examples and improve coverage.
- The retrieval budget is adapted at inference time without additional training.

### URASE: Semantic Execution Consistency

URASE adds a third objective to SQL selection in `po.py` and the call site in `main.py`. In addition to schema conformity and example consistency, it evaluates whether the executed result matches the semantic intent of the question.

Why it helps:
- Penalizes structurally valid but semantically wrong SQL.
- Uses execution metadata and question analysis instead of ground truth labels.
- Filters out candidates that look good syntactically but fail at the meaning level.

## 3. Net effect

- UARB improves which examples are retrieved.
- URASE improves which SQL candidate is selected.
- Together they reduce retrieval noise and semantic selection errors without retraining the base models.
