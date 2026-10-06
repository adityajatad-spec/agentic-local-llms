# Can Adaptive Planning, Tool Use, Memory, and Self-Verification Amplify Local LLMs on Complex Reasoning?

Code, prompts, per-problem results, and analysis for the paper of the same name. Every number in the paper is regenerated from the files in this repository by the scripts in `analysis/scripts/`.

## Contents

| Path | What it is |
|---|---|
| `agent.py` | All methods: CoT, self-consistency, Self-Refine, reflection memory, ReAct (print-only, REPL, native function calling), always-on pipeline, adaptive controller, thinking mode, gated thinking. All prompts are in this file. |
| `llm.py` | vLLM backends (offline engine and OpenAI-compatible server for native tool calls), a request batcher, and a CPU mock for smoke tests. |
| `data.py` | MATH-500, GSM8K, and GSM-Hard loaders with disjoint held-out (memory/calibration) and test slices, plus answer normalization. |
| `train.py` | Entry point: Modal job definition, per-method checkpointing, metrics, and a CPU smoke test. |
| `results/<run>/` | `metrics.json` (summary per method) and `per_example.jsonl` (every problem, answer, tokens, tool calls, and truncated traces). |
| `results/rebuilt-*` | Adaptive controller and gate→thinking rebuilt from independent self-consistency samples (see below). |
| `run_log/experiments.jsonl` | Log of all 28 runs, including failed and cancelled ones, with the reason. |
| `analysis/` | Scripts, generated tables (`tables/*.tex`), figures (`figures/*.pdf`), and `stats.json`. |

## Reproducing

```bash
pip install -r requirements.txt           # vllm is only needed inside the Modal image
python train.py --smoke_test              # CPU check of every method with a mock model

# GPU runs on Modal (A100). Gated models need a Modal secret named "huggingface-secret" with HF_TOKEN.
modal run --detach train.py --model Qwen/Qwen3-8B --task math --n 300 --n-mem 100 \
  --methods direct,cot,self_refine,reflexion,react,always_on,adaptive --output-dir results/run-XXX
```

The commands and settings for each logged run are in `run_log/experiments.jsonl` and each run's `metrics.json` (`meta`).

To regenerate every table, figure, and statistic from the stored results (no GPU needed):

```bash
python analysis/scripts/failure_taxonomy.py run-007 run-008 run-014 run-015 run-017 run-018 run-023 run-024 rebuilt-qwen@run-007 rebuilt-llama@run-017
python analysis/scripts/compute_stats.py
python analysis/scripts/empty_observations.py
cd analysis/scripts && for f in fig_*.py; do python $f; done
```

## Which run feeds which result

| Paper result | Runs |
|---|---|
| Qwen3-8B main table | 007 (original methods), 014 (REPL and native tools), 018 (thinking), 023 (self-consistency), rebuilt-qwen (adaptive, gate→thinking) |
| Llama-3.1-8B main table | 017, 015, 024, rebuilt-llama |
| Verifier ablation | 008 |
| Thinking mode seeds | 018, 026, 027 |
| GSM-Hard | 016, 025, rebuilt-gsmhard |
| GSM8K (appendix) | 004 |

## Note on sampling seeds

Early runs reused one per-request seed for every sampled request, which correlated repeated samples. The harness now gives each sampled request its own seed. Self-consistency was re-run (runs 023–025), and the adaptive controller and gate→thinking were rebuilt with `analysis/scripts/rebuild_gating.py`: gate decisions come from the first samples of those runs, and escalations come from clean source runs of the escalation target. Section 4 of the paper describes the procedure.

## License

Code is released under the MIT License (`LICENSE`). Results, run logs, tables, and figures are released under CC BY 4.0 (`LICENSE-DATA`). Model outputs remain subject to the Qwen3 and Llama 3.1 model licenses.
