"""LLM backends: a mock for CPU smoke tests and vLLM for real runs, plus a thread-safe batcher."""
import itertools
import queue
import re
import threading
from dataclasses import dataclass


@dataclass
class Gen:
    text: str
    n_in: int
    n_out: int
    mean_logprob: float = 0.0


@dataclass
class ChatOut:
    text: str
    tool_calls: list  # [(call_id, function_name, arguments_json_str)]
    n_in: int
    n_out: int


class MockLLM:
    """Deterministic stand-in so the whole pipeline can be exercised on CPU without a model."""

    def generate_batch(self, prompts, max_tokens=512, temperature=0.0, stop=None, think=False):
        return [self._one(p, think) for p in prompts]

    def _one(self, p, think=False):
        if think:
            t = "<think>\nLet me work through it.\n</think>\n\nFinal Answer: 42"
        elif "Write a Python program" in p:
            t = "```python\nprint(42)\n```"
        elif "Is this answer correct" in p:
            t = "Yes"
        elif "Reflect briefly" in p:
            t = "Double-check the arithmetic and units."
        elif "Make a short numbered plan" in p:
            t = "1. Identify quantities. 2. Compute. 3. Answer."
        elif p.endswith("Thought:") and "Observation:" not in p.split("Question:")[-1]:  # first ReAct step
            t = "Thought: compute it.\nAction:\n```python\nprint(42)\n```"
        else:
            t = "Reasoning...\nFinal Answer: 42"
        return Gen(t, max(1, len(p) // 4), max(1, len(t) // 4), -0.2)

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=1024, temperature=0.0):
        n_in = sum(len(str(m.get("content") or "")) for m in messages) // 4
        if messages[-1]["role"] == "tool" or tool_choice == "none" or not tools:
            return ChatOut("Done.\nFinal Answer: 42", [], n_in, 8)
        return ChatOut("", [("call_1", "run_python", '{"code": "x = 40\\nx + 2"}')], n_in, 12)


class VLLMBackend:
    def __init__(self, model_name, max_model_len=8192, seed=42):
        from vllm import LLM
        self.llm = LLM(model=model_name, max_model_len=max_model_len, gpu_memory_utilization=0.90, seed=seed)
        self.tok = self.llm.get_tokenizer()
        # Every sampled request gets its own seed. A shared per-request seed makes repeated samples of the same
        # prompt identical or highly correlated (this broke self-consistency and gate agreement in earlier runs).
        self._seeds = itertools.count(seed * 1_000_003)

    def _chat(self, p, think=False):
        msgs = [{"role": "user", "content": p}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate_batch(self, prompts, max_tokens=512, temperature=0.0, stop=None, think=False):
        from vllm import SamplingParams
        def params():
            if think:  # Qwen3 recommended thinking-mode sampling (greedy causes repetition loops)
                return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, logprobs=1,
                                      seed=next(self._seeds), stop=stop)
            if temperature > 0:
                return SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, logprobs=1,
                                      seed=next(self._seeds), stop=stop)
            return SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1, stop=stop)  # greedy: no seed needed
        sps = [params() for _ in prompts]
        outs = self.llm.generate([self._chat(p, think) for p in prompts], sps, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            n = max(1, len(c.token_ids))
            lp = (c.cumulative_logprob / n) if c.cumulative_logprob is not None else 0.0
            res.append(Gen(c.text, len(o.prompt_token_ids), len(c.token_ids), lp))
        return res


class ServerLLM:
    """vLLM OpenAI-compatible server in a subprocess; needed for native function calling via vLLM's tool parsers.

    Thread-safe: the server batches concurrent requests itself, so no BatchingLLM wrapper is used.
    """

    def __init__(self, model, max_model_len=8192, tool_parser="hermes", port=8000, log="/tmp/vllm_server.log", seed=42):
        import subprocess
        import time
        import urllib.request
        cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", str(max_model_len), "--seed", str(seed),
               "--gpu-memory-utilization", "0.90", "--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        self.log = log
        self.proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("vLLM server exited during startup:\n" + open(log).read()[-3000:])
            try:
                if urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5).status == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 1500:
                raise RuntimeError("vLLM server did not become healthy:\n" + open(log).read()[-3000:])
            time.sleep(5)
        print(f"vLLM server ready after {time.time() - t0:.0f}s (tool parser: {tool_parser})")
        from openai import OpenAI
        self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="local", timeout=1800, max_retries=2)
        self.model = model
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}
        self._seeds, self._lock = itertools.count(seed * 1_000_003), threading.Lock()

    def _create(self, messages, max_tokens, temperature, **kw):
        if temperature > 0:  # independent samples: one seed per sampled request
            with self._lock:
                kw["seed"] = next(self._seeds)
        return self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0, extra_body=self.extra, **kw)

    def generate(self, prompt, max_tokens=512, temperature=0.0, stop=None, think=False):
        if think:
            raise NotImplementedError("thinking mode is only wired for the offline backend")
        kw = {"logprobs": True}
        if stop:
            kw["stop"] = stop
        r = self._create([{"role": "user", "content": prompt}], max_tokens, temperature, **kw)
        c = r.choices[0]
        lps = [t.logprob for t in (c.logprobs.content or [])] if c.logprobs else []
        return Gen(c.message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens,
                   sum(lps) / len(lps) if lps else 0.0)

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=1024, temperature=0.0):
        kw = {"tools": tools, "tool_choice": tool_choice} if tools else {}
        r = self._create(messages, max_tokens, temperature, **kw)
        m = r.choices[0].message
        calls = [(tc.id, tc.function.name, tc.function.arguments) for tc in (m.tool_calls or [])]
        return ChatOut(m.content or "", calls, r.usage.prompt_tokens, r.usage.completion_tokens)

    def close(self):
        self.proc.terminate()


class BatchingLLM:
    """Collects generate() calls from many worker threads into batches for one backend call.

    Requests are grouped by (max_tokens, temperature) because SamplingParams is per-batch here.
    """

    def __init__(self, backend, max_batch=64, wait_s=0.02):
        self.backend, self.max_batch, self.wait_s = backend, max_batch, wait_s
        self.q = queue.Queue()
        self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def generate(self, prompt, max_tokens=512, temperature=0.0, stop=None, think=False):
        ev = threading.Event()
        box = {}
        self.q.put((prompt, max_tokens, temperature, tuple(stop) if stop else None, think, ev, box))
        ev.wait()
        if "err" in box:
            raise box["err"]
        return box["res"]

    def chat(self, *args, **kwargs):
        return self.backend.chat(*args, **kwargs)  # only the mock supports chat on the offline path

    def close(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                items = [self.q.get(timeout=0.1)]
            except queue.Empty:
                continue
            try:
                while len(items) < self.max_batch:
                    items.append(self.q.get(timeout=self.wait_s))
            except queue.Empty:
                pass
            groups = {}
            for it in items:
                groups.setdefault((it[1], it[2], it[3], it[4]), []).append(it)
            for (mt, temp, stp, thk), grp in groups.items():
                try:
                    res = self.backend.generate_batch([g[0] for g in grp], max_tokens=mt, temperature=temp,
                                                      stop=list(stp) if stp else None, think=thk)
                    for g, r in zip(grp, res):
                        g[6]["res"] = r
                except Exception as e:  # propagate to waiting threads instead of hanging them
                    for g in grp:
                        g[6]["err"] = e
                for g in grp:
                    g[5].set()


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not nums:
        return None
    s = nums[-1].replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None
