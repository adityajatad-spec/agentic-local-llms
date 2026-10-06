"""Entry point. Inference-only agent experiments (no training).

Local smoke test (CPU, mock LLM):   python train.py --smoke_test
Cloud run (Modal, vLLM, A100):      modal run --detach train.py --methods direct,cot,react --n 300 --output-dir results/run-XXX
  Partial results are checkpointed to the Modal volume after every method:
  modal volume get rstack-hf-cache results/run-XXX results/run-XXX
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import modal

SEED = 42
MODEL = "Qwen/Qwen3-8B"

app = modal.App("rstack-experiment")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "datasets", "numpy", "sympy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})  # no nvcc in slim image
    .add_local_python_source("llm", "data", "agent")
)
hf_cache = modal.Volume.from_name("rstack-hf-cache", create_if_missing=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)  # no torch.cuda.* here: it would init CUDA in the parent and break vLLM's fork
    except ImportError:
        pass


def bootstrap_ci(vals, iters=1000, seed=SEED):
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = [rng.choice(a, len(a), replace=True).mean() for _ in range(iters)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows):
    acc = [r["correct"] for r in rows]
    toks = [r["n_in"] + r["n_out"] for r in rows]
    flagged = [r for r in rows if r["flagged"] is not None and r["cand_correct"] is not None]
    wrong = [r for r in flagged if not r["cand_correct"]]
    tp = sum(1 for r in wrong if r["flagged"])
    fp = sum(1 for r in flagged if r["cand_correct"] and r["flagged"])
    s = {
        "n": len(rows), "accuracy": float(np.mean(acc)) if acc else 0.0, "acc_ci95": bootstrap_ci(acc),
        "tokens_per_problem": float(np.mean(toks)) if toks else 0.0,
        "latency_s_mean": float(np.mean([r["latency_s"] for r in rows])) if rows else 0.0,
        "tool_validity": (sum(r["tool_ok"] for r in rows) / max(1, sum(r["tool_calls"] for r in rows))),
        "escalation_rate": float(np.mean([r["escalated"] for r in rows])) if rows else 0.0,
    }
    s["acc_per_1k_tokens"] = s["accuracy"] / max(1e-9, s["tokens_per_problem"] / 1000)
    if flagged:
        s["verifier_recall"] = tp / len(wrong) if wrong else None
        s["verifier_false_reject_rate"] = fp / max(1, sum(1 for r in flagged if r["cand_correct"]))
    return s


def run_experiments(backend_name, methods, verifier, task, n, n_mem, smoke, data_dir, workers, model=MODEL,
                    checkpoint=None, seed=SEED):
    import agent
    from agent import METHODS, NEEDS_MEMORY, Ctx, build_memory
    from data import em_f1, load_gsm8k, load_gsmhard, load_hotpot, load_math500, synthetic_gsm
    same = lambda a, b: agent.same(a, b)
    agent.set_task("gsm8k" if smoke else task)
    from llm import BatchingLLM, MockLLM, ServerLLM, VLLMBackend

    set_seed()
    print(f"Data directory: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Contents: {os.listdir(data_dir)}")
    assert os.path.exists(data_dir), f"Data dir not found: {data_dir}"

    if smoke:
        test, mem_ex = synthetic_gsm(8), synthetic_gsm(4)
    elif task == "math":
        test, mem_ex = load_math500(n, n_mem, SEED, data_dir)
    elif task == "gsmhard":
        test, mem_ex = load_gsmhard(n, n_mem, SEED, data_dir)
    elif task == "gsm8k":
        test = load_gsm8k("test", n, SEED, data_dir)
        mem_ex = load_gsm8k("train", n_mem, SEED, data_dir)
    else:
        test, mem_ex = load_hotpot(n, SEED, data_dir), []
    print(f"Loaded {len(test)} test examples, {len(mem_ex)} memory examples")

    max_len = 20480 if any(m in agent.THINK_METHODS for m in methods) else 8192  # thinking needs a 16k budget
    if backend_name == "mock":
        llm = BatchingLLM(MockLLM())
    elif any(m in agent.NATIVE_METHODS for m in methods):  # native tool calls need the OpenAI-compatible server
        parser = "llama3_json" if "llama" in model.lower() else "hermes"
        llm = ServerLLM(model, max_model_len=max_len, tool_parser=parser, seed=seed)
    else:
        llm = BatchingLLM(VLLMBackend(model, max_model_len=max_len, seed=seed), max_batch=max(64, workers))
    memory = None
    if any(m in NEEDS_MEMORY for m in methods) and mem_ex:
        with ThreadPoolExecutor(1) as ex:
            memory = ex.submit(build_memory, llm, mem_ex).result()
        print(f"Memory built: {len(memory.items)} reflections")

    results = {}
    for m in methods:
        fn = METHODS[m]

        def solve(ex, fn=fn, m=m):
            ctx = Ctx()
            t0 = time.time()
            try:
                if task == "hotpot":
                    from llm import Gen  # noqa: F401
                    q = f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\n"
                    p = q + ("Think step by step, then end with 'Final Answer: <short answer>'.\n" if m == "cot" else "Answer in a few words.\nFinal Answer:")
                    txt = ctx.gen(llm, p, 300 if m == "cot" else 24).text
                    pred = txt.split("Final Answer:")[-1].strip().split("\n")[0]
                    em, f1 = em_f1(pred, ex["answer"])
                    correct, extra = em, {"f1": f1}
                else:
                    pred = fn(llm, ctx, ex, verifier=verifier, memory=memory)
                    correct, extra = float(same(pred, ex["answer"])), {}
            except Exception as e:
                pred, correct, extra = None, 0.0, {"error": repr(e)[:200]}
            return {"id": ex["id"], "method": m, "pred": pred, "gold": ex["answer"], "correct": correct,
                    "n_in": ctx.n_in, "n_out": ctx.n_out, "calls": ctx.calls, "tool_calls": ctx.tool_calls,
                    "tool_ok": ctx.tool_ok, "flagged": ctx.flagged, "escalated": ctx.escalated,
                    "info": ctx.info, "trace": ctx.trace, "cand_correct": None if ctx.cand is None else float(same(ctx.cand, ex["answer"])),
                    "latency_s": time.time() - t0, **extra}

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(solve, test))
        s = summarize(rows)
        s["wall_s"] = time.time() - t0
        results[m] = {"summary": s, "rows": rows}
        if checkpoint:
            checkpoint(results)  # persist after every method so a disconnect loses at most one method
        print(f"{m:12s} | acc {s['accuracy']:.3f} {s['acc_ci95']} | tok/prob {s['tokens_per_problem']:.0f} | wall {s['wall_s']:.0f}s")
    llm.close()
    return results


# The "huggingface-secret" Modal secret holds the HF token needed for gated models such as meta-llama/*.
hf_secret = modal.Secret.from_name("huggingface-secret")
TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
              "HUGGINGFACE_API_KEY", "HF_API_KEY", "HF_KEY", "HUGGINGFACE_KEY")


def _normalize_hf_token():
    """Expose the token as HF_TOKEN whatever key name it was stored under. Returns the key name found (never the value)."""
    for k in TOKEN_KEYS:
        if os.environ.get(k):
            os.environ["HF_TOKEN"] = os.environ[k]
            return k
    return None


@app.function(image=image, secrets=[hf_secret], timeout=300)
def check_hf(model: str = MODEL):
    """CPU-only preflight: is a token present, and does it grant access to the (gated) model?"""
    from huggingface_hub import hf_hub_download
    key = _normalize_hf_token()
    try:
        hf_hub_download(model, "config.json", token=os.environ.get("HF_TOKEN"))
        ok, err = True, None
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
    res = {"token_key_found": key, "access_ok": ok, "error": err}
    print(res)
    return res


@app.function(image=image, gpu="A100-80GB", timeout=4 * 3600, volumes={"/cache": hf_cache}, secrets=[hf_secret])
def run_remote(methods, verifier, task, n, n_mem, workers, model=MODEL, tag="latest", meta=None, seed=SEED):
    _normalize_hf_token()
    os.environ["HF_HOME"] = "/cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    ckpt_dir = f"/cache/results/{tag}"

    def checkpoint(results):
        save_outputs(results, ckpt_dir, {**(meta or {}), "model": model, "task": task, "n": n, "partial": True})
        hf_cache.commit()

    out = run_experiments("vllm", methods, verifier, task, n, n_mem, False, "/cache/data", workers, model=model,
                          checkpoint=checkpoint, seed=seed)
    hf_cache.commit()
    return out


def save_outputs(results, output_dir, meta):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {"meta": meta, "methods": {m: r["summary"] for m, r in results.items()}}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "per_example.jsonl"), "w") as f:
        for r in results.values():
            for row in r["rows"]:
                f.write(json.dumps(row) + "\n")
    return metrics


@app.local_entrypoint()
def main(methods: str = "direct,cot,react", verifier: str = "grounded", task: str = "gsm8k",
         n: int = 300, n_mem: int = 100, workers: int = 48, output_dir: str = "", hypothesis: str = "",
         model: str = MODEL, seed: int = SEED):
    ms = methods.split(",")
    t0 = time.time()
    out = output_dir or "results/latest"
    res = run_remote.remote(ms, verifier, task, n, n_mem, workers, model, os.path.basename(out.rstrip("/")),
                            {"hypothesis": hypothesis, "verifier": verifier, "seed": seed, "sampling": "per-request seeds"}, seed)
    meta = {"model": model, "task": task, "n": n, "verifier": verifier, "seed": seed, "sampling": "per-request seeds",
            "hypothesis": hypothesis,
            "duration_s": time.time() - t0}
    metrics = save_outputs(res, out, meta)
    print(json.dumps(metrics["methods"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--methods", type=str, default="direct,cot,react,self_refine,reflexion,always_on,adaptive")
    parser.add_argument("--verifier", type=str, default="grounded", choices=["grounded", "intrinsic", "none"])
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "math", "gsmhard", "hotpot"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not args.smoke_test:
        raise SystemExit("For real runs use: modal run train.py --methods ... (this entry only runs the CPU smoke test)")
    res = run_experiments("mock", args.methods.split(","), args.verifier, "gsm8k", 8, 4, True, args.data_dir, 4)
    save_outputs(res, args.output_dir, {"smoke_test": True})
    print(f"Smoke test OK. Outputs in {args.output_dir}")
