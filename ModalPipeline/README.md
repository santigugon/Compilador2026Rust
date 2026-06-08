# Qwen3-Coder × TritonBench Modal Evaluation Pipeline

Modal-based experiment runner for **Qwen3-Coder** checkpoints (BF16, AWQ, GPTQ) evaluated on **TritonBench-T** via the [TritonBench4Modal](https://github.com/salvahin/TritonBench4Modal) workflow.

## HOW TO RUN AN EXPERIMENT:

### PUSH SECRETS

```bash
python scripts/sync_secrets_from_env.py
```

```bash
source .venv/bin/activate
```

```bash
modal run --detach run_experiment.py \
  --config configs/qwen3_coder_experiment.yaml \
  --concurrency 10
```

## Secrets (`.env` → Modal)

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN=hf_...

pip install -r requirements.txt   # includes python-dotenv
python scripts/sync_secrets_from_env.py   # push .env → Modal secret hf-token
```

Each `modal run` **auto-syncs** from `.env` by default (`SYNC_SECRETS_ON_RUN=1`).

Details: [docs/SECRETS.md](docs/SECRETS.md). No OpenAI/NVIDIA API keys needed.

## Quick start

```bash
cd ModalPipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup

# Preview job plan -- use the venv Python (system python3 lacks PyYAML)
./scripts/run_local.sh --config configs/qwen3_coder_experiment.yaml --dry-run
# or: .venv/bin/python run_experiment.py --config configs/qwen3_coder_experiment.yaml --dry-run

# Run on Modal (requires hf-token secret)
modal run run_experiment.py --config configs/qwen3_coder_experiment.yaml

# Export artifacts
./scripts/export_results.sh qwen3_coder_tb_v1 ./local_results/qwen3_coder_tb_v1
```

## Safe smoke ramp

Start with one quantized model, one TritonBench item, no grammar, and a short vLLM
context. This avoids vLLM KV-cache startup failures on smaller serve GPUs.

```bash
# Preview exactly one Modal job.
modal run run_experiment.py \
  --config configs/qwen3_coder_smoke.yaml \
  --models qwen3_coder_30b_awq_4bit \
  --dry-run

# Run the same one-job smoke test and write results to the Modal volume.
modal run run_experiment.py \
  --config configs/qwen3_coder_smoke.yaml \
  --models qwen3_coder_30b_awq_4bit

# Artifacts are auto-exported after each completed job. Manual export is still available.
./scripts/export_results.sh qwen3_coder_smoke_v1 ./local_results/qwen3_coder_smoke_v1
```

If the smoke run passes, scale one dimension at a time:

- Increase `benchmark.limit` in `configs/qwen3_coder_smoke.yaml` from `1` to `2`, `5`, then larger values.
- Increase `vllm.max_model_len` only when needed for longer prompts/output.
- Increase `matrix.max_output_tokens` after the model can boot and write predictions.
- Add `xgrammar_on` under `matrix.decoding_modes` when the no-grammar path is stable.
- Add more model names with `--models name_a name_b`, or omit `--models` to run all enabled models.

## Results and resume

Modal writes experiment artifacts to the `qwen3-coder-results` volume under
`/{experiment_id}`. The runner automatically downloads that volume after each
completed condition and after final aggregation into:

```text
local_results/{experiment_id}/seed_000/
```

For multi-seed runs, each seed gets a matching `seed_###` directory. Manual
export uses the same layout:

```bash
./scripts/export_results.sh qwen3_coder_smoke_v1 ./local_results/qwen3_coder_smoke_v1 0
```

Completed Modal conditions write checkpoint files under
`/{experiment_id}/checkpoints/`. By default, reruns skip completed checkpoints,
so you can restart after an interruption and continue with the next unfinished
model/condition. Use `--no-resume` only when you intentionally want to rerun
everything.

If one Modal job fails before it can complete, the runner records it under
`/{experiment_id}/failed_checkpoints/`, auto-exports whatever is available, and
still finalizes aggregate tables for successful jobs. Use `--fail-fast` only
when you want one failed model startup to stop the whole run.

Run multiple Modal jobs concurrently with `--concurrency`. For example, this
keeps up to 8 model/condition jobs in flight:

```bash
modal run run_experiment.py \
  --config configs/qwen3_coder_experiment.yaml \
  --concurrency 8
```

If the config only expands to 5 jobs, concurrency 8 launches those 5. Larger
matrices keep at most 8 running at once.

## Configuration

- **Models**: [`configs/hf_models.yaml`](configs/hf_models.yaml) — edit `hf_id`, `enabled`, `benchmark_limit` per model.
- **Experiment**: [`configs/qwen3_coder_experiment.yaml`](configs/qwen3_coder_experiment.yaml) — matrix, `benchmark.limit`, `mode: pilot|full`.

## Outputs

Results land on Modal volume `qwen3-coder-results` under `/results/{experiment_id}/`:

- Per-attempt tree under `models/{model_name}/{quantization}/...`
- `predictions/{condition_id}.jsonl`
- `attempts.parquet`, aggregate CSVs after `finalize_experiment`

## Architecture

- **Generation**: vLLM on GPU (`image_serve`), optional XGrammar JSON schema per output profile.
- **Evaluation**: TritonBench call_acc → exe_acc → perf (`image_eval`).
- **MiniTriton JSON profile**: secondary; validated with `mini_triton_parser` from sibling `Compiler/` crate.

## Pilot vs full

`mode: pilot` in experiment YAML keeps a small matrix (see `expand_matrix` in `pipeline/config_loader.py`). Set `mode: full` for the complete factorial from the research spec.

## Analysis

```bash
python scripts/analyze_recommendation.py ./local_results/qwen3_coder_tb_v1
```
