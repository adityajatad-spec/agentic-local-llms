# Can Adaptive Planning, Tool Use, Memory, and Self-Verification Amplify Local LLMs on Complex Reasoning?

**Aditya Sopan Jatad**, Independent Researcher · [aditya.jatad@gmail.com](mailto:aditya.jatad@gmail.com)

📄 **Paper:** [`paper/agentic-local-llms-arxiv.pdf`](paper/agentic-local-llms-arxiv.pdf)

This repository holds the complete research artifact for the paper: the agent harness and every prompt, per-problem outputs for all runs used in the paper, the full run log (failed runs included), and the scripts that regenerate every table, figure, and statistic. No number in the paper is entered by hand. All of them are computed from the files here.

---

## Table of contents

1. [Summary](#summary)
2. [Key results](#key-results)
3. [Repository layout](#repository-layout)
4. [Methods implemented](#methods-implemented)
5. [Experimental setup](#experimental-setup)
6. [Installation](#installation)
7. [Reproducing the analysis (no GPU)](#reproducing-the-analysis-no-gpu)
8. [Re-running the experiments (GPU)](#re-running-the-experiments-gpu)
9. [Which run feeds which result](#which-run-feeds-which-result)
10. [Data formats](#data-formats)
11. [Rebuilt gated methods](#rebuilt-gated-methods)
12. [Run index](#run-index)
13. [Known issues and corrections](#known-issues-and-corrections)
14. [Limitations](#limitations)
15. [Citation](#citation)
16. [License](#license)

---

## Summary

Agentic scaffolds that wrap a language model with **planning**, **tool use**, **memory**, and **self-verification** are usually built and tested with frontier API models. This study asks whether the same components help **local 8B models** on competition mathematics.

- **Models:** Qwen3-8B and Llama-3.1-8B-Instruct, served with vLLM, with no fine-tuning.
- **Benchmarks:** MATH-500 (main), GSM-Hard (large-number arithmetic), and GSM8K (appendix).
- **Protocol:** each method is compared with chain-of-thought (CoT) from the same run, using exact McNemar tests on per-problem correctness with Holm correction. All costs are reported in tokens per problem.

**Findings**

1. **External scaffolding did not amplify either model.** The best external scaffolds tie CoT. The full pipeline and the adaptive controller are significantly worse.
2. **Internal test-time compute did.** Qwen3's thinking mode gains 10.3 points over CoT, consistently across three seeds, and most of the gain is on the hardest problems.
3. **The losses have concrete mechanisms:**
   - a code-based verifier that overrides correct answers;
   - tool semantics the model misjudges;
   - agent loops that never terminate.
4. **Adaptivity helps only when it routes to the right resource.** A disagreement gate finds hard problems reliably. Escalating them to tools hurts, while escalating them to thinking mode matches always-thinking, and a cheaper gate saves 19% of tokens.

---

## Key results

### MATH-500, 300 test problems

Δ is against CoT from the same run. Significance is Holm-corrected: \* p < 0.05, n.s. not significant. Tokens are prompt plus generated tokens per problem.

| Method | Qwen3-8B | Δ | Tokens | Llama-3.1-8B | Δ | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Direct answer | 19.3 | −60.3\* | 127 | 10.7 | −38.0\* | 125 |
| **Chain-of-thought (CoT)** | **79.7** | — | 937 | **48.7** | — | 800 |
| Self-Refine | 79.7 | +0.0 n.s. | 4,280 | 48.0 | −0.7 n.s. | 4,083 |
| Reflection memory | 80.0 | +0.3 n.s. | 1,015 | 44.3 | −4.3 n.s. | 1,180 |
| ReAct, print-only tool | 68.3 | −11.3\* | 1,709 | 4.7 | −44.0\* | 9,160 |
| ReAct, REPL tool | 75.3 | −3.0 n.s. | 1,934 | 21.3 | −30.7\* | 11,466 |
| ReAct, native function calls | 73.3 | −5.0 n.s. | 1,951 | 33.0 | −19.0\* | 1,504 |
| Always-on pipeline | 57.7 | −22.0\* | 3,201 | 35.3 | −13.3\* | 10,010 |
| Always-on, native calls | 61.0 | −17.3\* | 3,404 | 38.0 | −14.0\* | 2,883 |
| Adaptive controller | 73.3 | −6.3\* | 4,071 | 49.3 | +0.7 n.s. | 9,520 |
| Self-consistency@5 | 83.0 | +3.0 n.s. | 4,409 | 56.3 | +6.3 n.s. | 3,743 |
| **Thinking mode** | **90.0** | **+10.3\*** | 4,457 | — | — | — |
| Gate → thinking | 90.0 | +10.3\* | 5,042 | — | — | — |
| Gate (2 agree) → thinking | 89.7 | — | 3,624 | — | — | — |

### Thinking mode over three seeds (Qwen3-8B)

| Seed | CoT | Thinking | Δ |
|---|---:|---:|---:|
| 1 (run 018) | 79.7 | 90.0 | +10.3\* |
| 2 (run 026) | 80.0 | 88.7 | +8.7\* |
| 3 (run 027) | 77.7 | 89.7 | +12.0\* |
| Mean ± s.d. | | 89.4 ± 0.7 | +10.3 ± 1.7 |

### Self-verification in the always-on pipeline (Qwen3-8B, run 008)

Accuracy is measured before and after the verifier.

| Verifier | Before | After | Fixed | Broke |
|---|---:|---:|---:|---:|
| None | 65.0 | 65.0 | — | — |
| Grounded (code re-derivation) | 66.7 | 59.3 | 15 | 37 |
| Intrinsic (self-critique) | 67.3 | 75.7 | 38 | 13 |

### GSM-Hard (Qwen3-8B, 300 problems)

| CoT | ReAct (REPL) | ReAct (native) | Always-on | Adaptive | Self-consistency@5 |
|---:|---:|---:|---:|---:|---:|
| 76.7 | 76.7 | 69.7\* | 79.7 n.s. | 79.3 n.s. | 77.0 n.s. |

Full tables with 95% confidence intervals, ×CoT ratios, and exact p-values are in [`analysis/tables/`](analysis/tables/) and [`analysis/stats.json`](analysis/stats.json).

---

## Repository layout

```
agentic-local-llms/
├── paper/
│   └── agentic-local-llms-arxiv.pdf   # the paper
├── agent.py            # all methods, prompts, tools, memory, verifiers, gates
├── llm.py              # vLLM backends (offline engine and OpenAI-compatible server), batcher, CPU mock
├── data.py             # MATH-500 / GSM8K / GSM-Hard loaders, held-out and test slices, answer normalization
├── train.py            # entry point: Modal job, per-method checkpointing, metrics, CPU smoke test
├── requirements.txt
├── results/
│   ├── run-XXX/        # metrics.json + per_example.jsonl for every run used in the paper
│   └── rebuilt-*/      # adaptive controller and gate→thinking rebuilt from independent samples
├── run_log/
│   └── experiments.jsonl   # all 28 runs, including failed and cancelled ones, with reasons
├── analysis/
│   ├── scripts/        # compute_stats.py, failure_taxonomy.py, rebuild_gating.py, empty_observations.py, fig_*.py
│   ├── tables/         # generated LaTeX tables used in the paper
│   ├── figures/        # generated figures (PDF and PNG)
│   ├── stats.json      # every statistic quoted in the paper
│   ├── failures.json   # failure taxonomy and accuracy by difficulty level
│   └── empty_observations.json
├── LICENSE             # MIT (code)
└── LICENSE-DATA        # CC BY 4.0 (results, logs, tables, figures)
```

---

## Methods implemented

All methods live in [`agent.py`](agent.py) and are selected by name with `--methods`.

| Name | Method | Notes |
|---|---|---|
| `direct` | Direct answer | Final answer only |
| `cot` | Chain-of-thought | Greedy; ends with `Final Answer:` |
| `sc5` | Self-consistency@5 | Majority vote over 5 samples at temperature 0.7; stores per-sample confidence and tokens |
| `self_refine` | Self-Refine | One round of critique and revision |
| `reflexion` | Reflection memory | Lessons from held-out failures, retrieved by similarity (top 3) |
| `react` | ReAct, print-only tool | Text format; stateless script; only printed output returned |
| `react_repl` | ReAct, REPL tool | Text format; stateful notebook-style tool; forced final answer at the step limit |
| `react_native` | ReAct, native function calls | The model's own tool-call format via vLLM parsers (Hermes for Qwen3, Llama-3 JSON for Llama) |
| `always_on` | Always-on pipeline | Plan → ReAct with memory → grounded verifier |
| `always_on_native` | Always-on, native calls | Same, with native ReAct |
| `always_on_intrinsic`, `always_on_none` | Verifier ablation | Intrinsic verifier, or no verifier |
| `adaptive` | Adaptive controller | 3 samples; accept if unanimous and confident, else escalate to the always-on pipeline |
| `think` | Thinking mode | Qwen3 native reasoning (temperature 0.6, top-p 0.95, top-k 20; 16,384-token budget) |
| `gate_think` | Gate → thinking | The same gate, escalating to thinking mode |

**Components**

- **Planner:** a short numbered plan written before solving.
- **Python tool:** three designs.
  - *print-only:* each block runs as a fresh script, and only printed output comes back;
  - *REPL:* variables persist, the value of a final bare expression is shown, and empty output is reported as `(no output)`;
  - *native:* the same REPL tool, exposed through the model's own function-calling format.
- **Memory:** one-sentence lessons written after wrong held-out attempts, with the gold answer used as feedback. The store is frozen before testing.
- **Verifier:**
  - *grounded:* an independent program re-derives the answer and replaces the candidate on disagreement;
  - *intrinsic:* a yes/no self-check, falling back to CoT on "No".
- **Gate:** three temperature-0.7 CoT samples must agree, and their mean token log-probability must exceed a threshold. The threshold is the 25th percentile on the held-out slice, with one sample per held-out problem.

---

## Experimental setup

| Item | Setting |
|---|---|
| Models | `Qwen/Qwen3-8B` (non-thinking unless stated) and `meta-llama/Llama-3.1-8B-Instruct`, bf16 |
| Serving | vLLM 0.31.0 on one NVIDIA A100 (40 GB or 80 GB) via [Modal](https://modal.com); torch 2.13.0, transformers 5.17.0, datasets 5.0.1 |
| Data split | MATH-500 shuffled once with seed 42: problems 1–100 form the held-out slice (memory and calibration), problems 101–400 the test set. GSM-Hard is split the same way. |
| Decoding | Greedy for single-sample methods; temperature 0.7 and top-p 0.95 for sampling; thinking uses temperature 0.6, top-p 0.95, top-k 20 |
| Budgets (MATH) | CoT 4,096 tokens; ReAct 1,024 tokens per step, 8 steps; thinking 16,384 tokens |
| Answer matching | Text after the last `Final Answer:` (or `\boxed{}`), LaTeX-normalized; exact string or numeric equality |
| Statistics | 95% bootstrap CIs; exact McNemar against CoT from the same run; Holm correction within each table |
| Seeds | Thinking mode: 3 seeds. Greedy methods are deterministic up to batch effects. Every sampled request uses its own seed derived from the run seed. |

---

## Installation

```bash
git clone https://github.com/adityajatad-spec/agentic-local-llms.git
cd agentic-local-llms
python -m pip install -r requirements.txt   # vllm is only needed inside the Modal image
```

Python 3.11 or later. The analysis scripts additionally need `scipy` (included in `requirements.txt`).

Check that everything runs, on CPU with a mock model and no downloads:

```bash
python train.py --smoke_test --output_dir /tmp/smoke
```

---

## Reproducing the analysis (no GPU)

Every table, figure, and number in the paper is regenerated from the stored results:

```bash
python analysis/scripts/failure_taxonomy.py run-007 run-008 run-014 run-015 run-017 run-018 run-023 run-024 \
       rebuilt-qwen@run-007 rebuilt-llama@run-017
python analysis/scripts/compute_stats.py        # writes analysis/stats.json and analysis/tables/*.tex
python analysis/scripts/empty_observations.py   # tool calls whose output was empty
cd analysis/scripts && for f in fig_*.py; do python "$f"; done   # writes analysis/figures/
```

To rebuild the gated methods from the independent self-consistency samples (see [Rebuilt gated methods](#rebuilt-gated-methods)):

```bash
python analysis/scripts/rebuild_gating.py run-023 math    rebuilt-qwen    adaptive=run-007:always_on gate_think=run-018:think
python analysis/scripts/rebuild_gating.py run-024 math    rebuilt-llama   adaptive=run-017:always_on
python analysis/scripts/rebuild_gating.py run-025 gsmhard rebuilt-gsmhard adaptive=run-016:always_on
```

---

## Re-running the experiments (GPU)

1. **Set up Modal:** `pip install modal`, then `modal token new`.
2. **Hugging Face token:** gated models (Llama) need a Modal secret named `huggingface-secret` with key `HF_TOKEN`:
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_...
   ```
3. **Launch a run.** It is detached, so it keeps running if your laptop disconnects, and each method is checkpointed to the Modal volume `rstack-hf-cache` under `results/<run>`.
   ```bash
   modal run --detach train.py --model Qwen/Qwen3-8B --task math --n 300 --n-mem 100 \
     --methods cot,self_refine,reflexion,react,always_on,adaptive --output-dir results/run-XXX
   ```

**Options**

| Flag | Meaning |
|---|---|
| `--model` | Hugging Face model id |
| `--task` | `math`, `gsmhard`, or `gsm8k` |
| `--n`, `--n-mem` | Test problems; held-out problems for memory and calibration |
| `--methods` | Comma-separated method names (see the table above) |
| `--verifier` | `grounded`, `intrinsic`, or `none` (for `always_on`) |
| `--workers` | Concurrent problems; use about 300 for sampling and thinking methods |
| `--seed` | Run seed; per-request sampling seeds are derived from it |

Runs that include `react_native` or `always_on_native` start vLLM's OpenAI-compatible server automatically. All other runs use the offline engine. The exact command settings of each logged run are in its `metrics.json` (`meta`) and in [`run_log/experiments.jsonl`](run_log/experiments.jsonl). Logged runs took 5 to 107 minutes each on one A100 (`meta.duration_s`).

To retrieve a checkpoint after a disconnect:

```bash
modal volume get rstack-hf-cache results/run-XXX results/
```

---

## Which run feeds which result

| Paper result | Runs |
|---|---|
| Qwen3-8B main table | 007 (original methods), 014 (REPL and native tools), 018 (thinking), 023 (self-consistency), `rebuilt-qwen` (adaptive, gate → thinking) |
| Llama-3.1-8B main table | 017, 015, 024, `rebuilt-llama` |
| Verifier ablation | 008 |
| Thinking-mode seeds | 018, 026, 027 |
| Gating table | 018, 023, `rebuilt-qwen` |
| GSM-Hard | 016, 025, `rebuilt-gsmhard` |
| Failure taxonomy, accuracy by level | 007, 014, 015, 017, 018, 023, 024, `rebuilt-*` |
| GSM8K (appendix) | 004 |

---

## Data formats

**`results/<run>/metrics.json`**

```json
{
  "meta":    {"model": "...", "task": "math", "n": 300, "seed": 42, "hypothesis": "...", "duration_s": 6437},
  "methods": {"cot": {"n": 300, "accuracy": 0.797, "acc_ci95": [0.75, 0.84], "tokens_per_problem": 937.0,
                      "latency_s_mean": 51.0, "tool_validity": 0.0, "escalation_rate": 0.0,
                      "acc_per_1k_tokens": 0.85, "wall_s": 353}}
}
```

Methods with a verifier also report `verifier_recall` and `verifier_false_reject_rate`.

**`results/<run>/per_example.jsonl`** has one line per problem and method:

| Field | Meaning |
|---|---|
| `id`, `method`, `gold`, `pred`, `correct` | Problem id, method, reference answer, prediction, 1.0 or 0.0 |
| `n_in`, `n_out`, `calls` | Prompt tokens, generated tokens, and model calls, summed over the problem |
| `tool_calls`, `tool_ok` | Tool executions and how many succeeded |
| `flagged`, `cand_correct` | Whether the verifier or refiner acted, and whether the pre-verification candidate was correct |
| `escalated` | Whether a gated method escalated |
| `info` | Method extras: gate answers and confidence, threshold, per-sample tokens, `think_truncated`, `forced_final` |
| `trace` | Per call: prompt tail (300 characters), output (head and tail if long), generated tokens |

---

## Rebuilt gated methods

Early runs reused one per-request sampling seed, which made repeated samples of the same prompt highly correlated. Only 110 of 300 problems had five distinct self-consistency samples. The harness now gives every sampled request its own seed: 299, 300, and 299 of 300 problems have five distinct samples in runs 023 to 025.

Rather than re-running the adaptive controller and gate → thinking end to end, [`rebuild_gating.py`](analysis/scripts/rebuild_gating.py) reconstructs them:

1. **Gate decision:** taken from the first three independent samples of the self-consistency run, using the threshold calibrated in the original run. That calibration used one sample per held-out problem, so the bug did not affect it.
2. **Accepted problems:** take the agreed answer.
3. **Escalated problems:** take the answer of the escalation target from an existing clean run on the same problems.
   - *Always-on pipeline:* it decodes greedily, so this is exactly the answer the controller would have produced.
   - *Thinking mode:* its sampled answer is an independent draw of the same procedure.
4. **Cost:** the gate's samples plus, for escalated problems only, the escalation target's tokens, both from the source runs.
5. **Reference CoT:** each rebuilt row is tested against the CoT of its escalation source run. That is 007 for Qwen, 017 for Llama, 018 for gate → thinking, and 016 for GSM-Hard.

---

## Run index

All 28 logged runs. Runs marked ✓ have their results in `results/`. Failed runs are kept in the log for transparency.

| Run | Status | In repo | Description |
|---|---|:---:|---|
| run-001 | failed |  | Baselines: direct vs CoT vs ReAct+python tool on Qwen3-8B, GSM8K-300 |
| run-002 | failed |  | Baselines: direct vs CoT vs ReAct+python tool on Qwen3-8B, GSM8K-300 (retry after CUDA fork fix) |
| run-003 | completed | ✓ | Baselines: direct vs CoT vs ReAct+python tool on Qwen3-8B, GSM8K-300 (single seed; direct truncated at 16 ... |
| run-004 | completed | ✓ | All 7 methods, Qwen3-8B GSM8K-300, grounded verifier |
| run-005 | failed | ✓ | All 7 methods, Qwen3-8B, MATH-500 (300 test) |
| run-006 | completed | ✓ | PILOT n=60 MATH-500 validating harness fixes |
| run-007 | completed | ✓ | MAIN: 7 methods, Qwen3-8B, MATH-500 (300 test + 100 held-out), validated harness |
| run-008 | completed | ✓ | Verifier ablation, always-on pipeline, same 300 MATH-500 problems, shared memory |
| run-009 | completed | ✓ | PILOT n=30 Llama-3.1-8B-Instruct MATH-500: harness validation |
| run-010 | failed |  | Llama-3.1-8B all 7 methods MATH-500 |
| run-011 | failed |  | Qwen3-8B gated routing (cot, sc5, think, gate_think) |
| run-012 | completed | ✓ | PILOT n=20 Qwen3-8B native tool calling (hermes) + stateful REPL tool |
| run-013 | completed | ✓ | PILOT n=20 Llama-3.1-8B native tool calling (llama3_json) + stateful REPL tool |
| run-014 | completed | ✓ | #3 native tool calling, Qwen3-8B MATH-500 (server backend) |
| run-015 | completed | ✓ | #3 native tool calling, Llama-3.1-8B MATH-500 |
| run-016 | completed | ✓ | #4 GSM-Hard Qwen3-8B (300 test + 100 held-out) |
| run-017 | completed | ✓ | Llama-3.1-8B all 7 methods MATH-500 (rerun of run-010; merged from 3 parallel detached jobs, A100-80GB) |
| run-018 | completed | ✓ | Qwen3-8B gated routing MATH-500 (rerun of run-011; merged from 3 jobs): cot .797 (974 tok), sc5 .820 (4467), ... |
| run-019 | failed |  | Seed-fix rerun (CoT, SC@5, thinking; Qwen3-8B) |
| run-019-pilot | completed |  | PILOT n=10: verify per-request sampling seeds |
| run-020 | failed |  | Seed-fix rerun (gate->thinking; Qwen3-8B) |
| run-021 | failed |  | Seed-fix rerun (adaptive; Qwen3-8B) |
| run-022 | failed |  | Seed-fix rerun (adaptive; Llama-3.1-8B) |
| run-023 | completed | ✓ | Seed-fix: CoT + SC@5 with independent per-request seeds, Qwen3-8B MATH-500 |
| run-024 | completed | ✓ | Seed-fix: CoT + SC@5 with independent seeds, Llama-3.1-8B MATH-500 |
| run-025 | completed | ✓ | Seed-fix: CoT + SC@5 with independent seeds, Qwen3-8B GSM-Hard |
| run-026 | completed | ✓ | Extra seed (2): CoT + thinking mode, Qwen3-8B MATH-500 |
| run-027 | completed | ✓ | Extra seed (3): CoT + thinking mode, Qwen3-8B MATH-500 |

---

## Known issues and corrections

Every issue below was found, fixed, and logged. The affected runs are marked in the run log, and the paper reports only corrected results.

| Issue | Effect | Fix |
|---|---|---|
| CUDA initialized before vLLM forked (runs 001–002) | Engine failed to start | No CUDA calls in the parent process; spawn workers |
| FlashInfer sampler needed `nvcc` | Engine failed at the first sample | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| ReAct accepted hallucinated observations (run 004) | 98% of problems never ran a tool | Stop generation at `Observation:` |
| Markdown-bold `Final Answer` and a 1,024-token cap (run 005) | Many answers lost on MATH | Robust extraction; 4,096-token cap |
| Print-only tool (paper Section 5.2) | 18.5% of observations empty | Stateful REPL tool and native calling, reported as variants |
| Shared sampling seed | Correlated samples | Per-request seeds; gated methods rebuilt (above) |
| Dropped client connection (runs 010–011) | Runs cancelled | `--detach` and per-method checkpoints |

---

## Limitations

- **Scope:** two 8B models, mathematical reasoning only, 300 test problems per benchmark. Intervals are about ±5 points, so small differences remain unresolved.
- **Seeds:** thinking mode has three seeds; self-consistency and the gates have one.
- **Harness dependence:** tool design and prompts affect results.
- **Reconstruction:** the gated methods are reconstructed rather than re-run end to end (see above).
- **Labels and matching:** GSM-Hard labels come from executed programs, and some are implausible. The answer matcher is conservative, equally for all methods.

---

## Citation

```bibtex
@misc{jatad2026agentic,
  title  = {Can Adaptive Planning, Tool Use, Memory, and Self-Verification Amplify Local Large Language Models on Complex Reasoning?},
  author = {Jatad, Aditya Sopan},
  year   = {2026},
  note   = {Code and data: https://github.com/adityajatad-spec/agentic-local-llms}
}
```

---

## License

- **Code:** [MIT License](LICENSE).
- **Results, run logs, tables, and figures:** [CC BY 4.0](LICENSE-DATA).
- **Model outputs:** generated by Qwen3-8B and Llama-3.1-8B-Instruct, and subject to their respective licenses.
