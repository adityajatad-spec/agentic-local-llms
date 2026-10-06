"""Dataset loading. Smoke mode uses a tiny synthetic arithmetic set (no network)."""
import random
import re


def synthetic_gsm(n=8):
    rng = random.Random(0)
    out = []
    for i in range(n):
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        ans = 42 if i % 2 == 0 else a + b
        q = f"Sam has {a} apples and buys {b} more. " + ("Then the total is multiplied so that it equals 42. " if i % 2 == 0 else "") + "How many apples?"
        out.append({"id": f"syn-{i}", "question": q, "answer": float(ans)})
    return out


def _gold_gsm(ans_text):
    return float(ans_text.split("####")[-1].strip().replace(",", ""))


def load_gsm8k(split, n, seed, cache_dir):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    return [{"id": f"gsm-{split}-{i}", "question": ds[i]["question"], "answer": _gold_gsm(ds[i]["answer"])} for i in idx]


def load_math500(n, n_mem, seed, cache_dir):
    """MATH-500 split into a held-out memory/calibration slice and a disjoint test slice (no leakage)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    mk = lambda i: {"id": f"math-{ds[i]['unique_id']}", "question": ds[i]["problem"], "answer": ds[i]["answer"]}
    mem = [mk(i) for i in idx[:n_mem]]
    test = [mk(i) for i in idx[n_mem:n_mem + n]]
    return test, mem


def load_gsmhard(n, n_mem, seed, cache_dir):
    """GSM-Hard (GSM8K with large numbers; labels from executing programs). Disjoint memory / test slices."""
    from datasets import load_dataset
    ds = load_dataset("reasoning-machines/gsm-hard", split="train", cache_dir=cache_dir)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    mk = lambda i: {"id": f"gsmhard-{i}", "question": ds[i]["input"], "answer": float(ds[i]["target"])}
    return [mk(i) for i in idx[n_mem:n_mem + n]], [mk(i) for i in idx[:n_mem]]


def load_hotpot(n, seed, cache_dir):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=cache_dir)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        ctx = "\n".join(f"[{t}] " + " ".join(s) for t, s in zip(r["context"]["title"], r["context"]["sentences"]))
        out.append({"id": f"hp-{r['id']}", "question": r["question"], "context": ctx, "answer": r["answer"]})
    return out


def norm_text(s):
    s = re.sub(r"\b(a|an|the)\b", " ", (s or "").lower())
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def em_f1(pred, gold):
    p, g = norm_text(pred).split(), norm_text(gold).split()
    em = float(p == g)
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not p or not g or common == 0:
        return em, 0.0
    pr, rc = common / len(p), common / len(g)
    return em, 2 * pr * rc / (pr + rc)
