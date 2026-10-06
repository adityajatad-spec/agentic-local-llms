"""Agent methods (baselines + the proposed components) over a generic `llm.generate` interface."""
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm import last_number

MAX_STEPS = 8
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


class Ctx:
    """Per-example accounting: tokens, LLM calls, tool calls, verifier outcome."""

    def __init__(self):
        self.n_in = self.n_out = self.calls = self.tool_calls = self.tool_ok = 0
        self.flagged = None       # did the verifier reject the first candidate?
        self.cand = None          # first candidate answer (before verification)
        self.escalated = False
        self.info = {}            # controller signals (agreement, confidence)
        self.trace = []           # (prompt tail, completion) per call, truncated, for failure analysis

    def gen(self, llm, prompt, max_tokens=512, temperature=0.0, stop=None, think=False):
        g = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop, think=think)
        self.n_in += g.n_in
        self.n_out += g.n_out
        self.calls += 1
        o = g.text if len(g.text) <= 1200 else g.text[:300] + " ...[cut]... " + g.text[-900:]
        self.trace.append({"p": prompt[-300:], "o": o, "n_out": g.n_out})
        return g

    def chat(self, llm, messages, tools=None, tool_choice="auto", max_tokens=1024):
        out = llm.chat(messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens)
        self.n_in += out.n_in
        self.n_out += out.n_out
        self.calls += 1
        o = out.text if len(out.text) <= 1200 else out.text[:300] + " ...[cut]... " + out.text[-900:]
        if out.tool_calls:
            o += " [TOOL_CALLS] " + json.dumps([c[2] for c in out.tool_calls])[:800]
        self.trace.append({"p": json.dumps(messages[-1])[-300:], "o": o, "n_out": out.n_out})
        return out


# ---------------------------------------------------------------- tools
def run_python(code, timeout=5):
    """Execute model-written code in a subprocess; return (ok, stdout_or_error)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return False, (r.stderr.strip().splitlines() or ["error"])[-1][:300]
        return True, r.stdout.strip()[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


REPL_TEMPLATE = """import ast, contextlib, io
G = {"__name__": "__main__"}
for _src in %r:  # replay earlier successful cells silently so variables persist
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(_src, "<cell>", "exec"), G)
        except Exception:
            pass
_tree = ast.parse(%r)
_last = _tree.body.pop() if _tree.body and isinstance(_tree.body[-1], ast.Expr) else None
exec(compile(_tree, "<cell>", "exec"), G)
if _last is not None:  # display the value of a final bare expression, like a notebook
    _v = eval(compile(ast.Expression(_last.value), "<cell>", "eval"), G)
    if _v is not None:
        print(repr(_v))
"""


def run_repl(history, code, timeout=10):
    """Stateful, notebook-like Python tool. `history` holds earlier successful cells for this problem."""
    ok, out = run_python(REPL_TEMPLATE % (history, code), timeout=timeout)
    if ok:
        history.append(code)
        out = out or "(no output)"
    return ok, out


def extract_code(text):
    m = CODE_RE.search(text or "")
    return m.group(1) if m else None


# ---- task hooks: GSM8K answers are floats, MATH answers are normalized strings
TASK = {"name": "gsm8k", "cot_tok": 512, "react_tok": 400, "think_tok": 8192}


def set_task(name):
    cot = {"math": 4096, "gsmhard": 1024}.get(name, 512)
    react = {"math": 1024, "gsmhard": 600}.get(name, 400)
    TASK.update(name=name, cot_tok=cot, react_tok=react, think_tok=16384 if name == "math" else 8192)


def _is_math():
    return TASK["name"] == "math"


def hint():
    return "<answer>" if _is_math() else "<number>"


def norm_math(s):
    s = str(s).strip().replace("$", "").strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", s, re.S)
    s = m.group(1) if m else s
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    for t in ("\\left", "\\right", "\\!", "\\,", "\\ ", "$", " ", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(t, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)          # \frac12 -> \frac{1}{2}
    s = re.sub(r"^(-?)(\d+)/(\d+)$", r"\1\\frac{\2}{\3}", s)     # 3/4 -> \frac{3}{4}
    return s.lower() if re.fullmatch(r"[A-Za-z ()]+", s) else s


def _math_val(s):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    try:
        if m:
            return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        return float(s.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _clean(text):
    return (text or "").replace("**", "").replace("__", "")


def extract_answer(text):
    text = _clean(text)
    hits = re.findall(r"final answer(?:[ \t]+is)?[ \t]*[:：]?[ \t]*(.*)", text, re.I)  # stay on one line
    if _is_math():
        if hits:
            line = hits[-1].strip()
            b = re.search(r"\\boxed\{(.*)\}", line)
            ans = norm_math(b.group(1) if b else line)
            if ans:
                return ans
        b = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        return norm_math(b[-1]) if b else None
    if hits:
        v = last_number(hits[-1])
        if v is not None:
            return v
    return last_number(text)


def from_text(text):
    """Parse a bare value (direct answers, tool stdout)."""
    if _is_math():
        t = _clean(text)
        if "\\boxed" in t or re.search(r"final answer", t, re.I):
            return extract_answer(t)
        lines = [l for l in t.strip().splitlines() if l.strip() and l.strip() not in ("$$", "\\[", "\\]")]
        return norm_math(lines[-1]) if lines else None
    return last_number(text)


def same(a, b):
    if a is None or b is None:
        return False
    if _is_math():
        na, nb = norm_math(a), norm_math(b)
        if na == nb:
            return True
        va, vb = _math_val(na), _math_val(nb)
        return va is not None and vb is not None and math.isclose(va, vb, rel_tol=1e-6, abs_tol=1e-6)
    rel = 1e-4 if TASK["name"] == "gsmhard" else 1e-6  # GSM-Hard answers are large floats; allow rounding
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-4)


def key(a):
    return norm_math(a) if _is_math() else round(a, 4)


def show(a):
    return norm_math(a) if _is_math() else f"{a:g}"


# ---------------------------------------------------------------- memory
def _bow(t):
    return Counter(re.findall(r"[a-z0-9]+", t.lower()))


def _cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


class Memory:
    """Frozen episodic store of reflections built from a held-out split (no test leakage)."""

    def __init__(self):
        self.items = []  # (question_bow, reflection)
        self.lp_thr = None

    def add(self, question, reflection):
        self.items.append((_bow(question), reflection))

    def retrieve(self, question, k=3):
        qb = _bow(question)
        scored = sorted(self.items, key=lambda it: -_cos(qb, it[0]))[:k]
        return [r for _, r in scored]

    def block(self, question, k=3):
        r = self.retrieve(question, k)
        return ("Lessons from similar past problems:\n- " + "\n- ".join(r) + "\n\n") if r else ""


def build_memory(llm, train_examples, memory_cls=Memory, workers=32):
    """Reflexion-style: attempt held-out problems, reflect on wrong ones using the gold label as env reward.

    Problems run in parallel so the batcher can group them; reflections are added in input order (deterministic).
    """
    def one(ex):
        ctx = Ctx()
        g = ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"])
        lp = ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"], temperature=0.7).mean_logprob  # match adaptive sampling
        refl = None
        if not same(extract_answer(g.text), ex["answer"]):
            r = ctx.gen(llm, f"Question: {ex['question']}\nA wrong attempt:\n{g.text}\n"
                             f"The correct answer is {show(ex['answer'])}. Reflect briefly (one sentence, general lesson, "
                             "do not state the answer):", 80)
            refl = r.text.strip().split("\n")[0]
        return ex["question"], lp, refl

    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, train_examples))
    mem = memory_cls()
    for q, _, refl in res:
        if refl is not None:
            mem.add(q, refl)
    lps = sorted(lp for _, lp, _ in res)
    if lps:  # escalation threshold from held-out CoT confidence (25th percentile), never from test data
        mem.lp_thr = lps[len(lps) // 4]
    return mem


# ---------------------------------------------------------------- prompts
def cot_prompt(q, mem_block=""):
    return f"{mem_block}Solve step by step. End with 'Final Answer: {hint()}'.\n\nQuestion: {q}\n"


def code_prompt(q):
    return f"Write a Python program that computes the answer to this problem. Print only the final {'answer in simplest exact form (use sympy if helpful)' if _is_math() else 'number'}.\n\nQuestion: {q}\n"


# ---------------------------------------------------------------- verifiers
def verify_grounded(llm, ctx, q, cand):
    """Re-derive the answer by executing model-written code. Returns (accepted, tool_value)."""
    g = ctx.gen(llm, code_prompt(q), TASK["react_tok"])
    code = extract_code(g.text)
    ctx.tool_calls += 1
    if code is None:
        return True, None  # cannot ground -> do not reject
    ok, out = run_python(code)
    ctx.tool_ok += int(ok)
    val = from_text(out) if ok else None
    if val is None:
        return True, None
    return same(val, cand), val


def verify_intrinsic(llm, ctx, q, cand_text):
    g = ctx.gen(llm, f"Question: {q}\nProposed solution:\n{cand_text}\n\nIs this answer correct? Reply Yes or No.", 5)
    return not g.text.strip().lower().startswith("no")


# ---------------------------------------------------------------- methods (GSM8K)
def m_direct(llm, ctx, ex, **kw):
    g = ctx.gen(llm, f"Answer with only the final {'answer' if _is_math() else 'number'}, no explanation.\n\nQuestion: {ex['question']}\nFinal Answer:", 64)
    return from_text(g.text)


def m_cot(llm, ctx, ex, **kw):
    return extract_answer(ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"]).text)


def m_react(llm, ctx, ex, plan="", mem_block="", tool="script", force_final=False, **kw):
    """Text-format ReAct. tool="script": stateless, print-only (original). tool="repl": stateful notebook-like tool."""
    tool_desc = ("followed by a ```python code block that prints its result" if tool == "script" else
                 "followed by a ```python code block; variables persist between actions and the value of a final "
                 "bare expression is shown")
    hdr = ("Solve the problem using Thought / Action / Observation steps.\n"
           "In each Thought, first reason step by step about what the problem asks. To compute, write 'Action:' "
           f"{tool_desc}. Check the Observation against your reasoning. "
           "When sure, write 'Final Answer: "+hint()+"'.\n\n"
           "Example:\nQuestion: A pen costs $3 and a book costs 4 times as much. Total for 2 pens and 1 book?\n"
           "Thought: The book costs 4*3=12. Two pens cost 6. I will verify with code.\n"
           "Action:\n```python\nprint(2*3 + 4*3)\n```\nObservation: 18\n"
           "Thought: Matches my reasoning.\nFinal Answer: 18\n\nNow solve:\n\n")
    prompt = f"{hdr}{mem_block}{plan}Question: {ex['question']}\n"
    history = []
    for _ in range(MAX_STEPS):
        g = ctx.gen(llm, prompt + "Thought:", TASK["react_tok"], stop=["Observation:"])
        text = "Thought:" + g.text.split("Observation:")[0]  # anything after is a hallucinated tool result
        code = extract_code(text)
        if code is None:
            return extract_answer(text) if re.search(r"Final Answer:", text, re.I) else from_text(text)
        ok, out = run_python(code) if tool == "script" else run_repl(history, code)
        ctx.tool_calls += 1
        ctx.tool_ok += int(ok)
        prompt += f"{text.split('Action:')[0]}Action:\n```python\n{code}```\nObservation: {out}\n"
    if not force_final:
        return None
    ctx.info = {**ctx.info, "forced_final": True}
    g = ctx.gen(llm, prompt + "Thought: I must stop using tools now and give my answer.\nFinal Answer:", 256)
    return extract_answer("Final Answer:" + g.text)


def m_react_repl(llm, ctx, ex, plan="", mem_block="", **kw):
    return m_react(llm, ctx, ex, plan=plan, mem_block=mem_block, tool="repl", force_final=True)


TOOLS = [{"type": "function", "function": {
    "name": "run_python",
    "description": ("Execute Python code in a persistent session and return its output. Printed output and the value "
                    "of a final bare expression are returned; variables persist between calls. sympy is installed."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code"}},
                   "required": ["code"]}}}]


def m_react_native(llm, ctx, ex, plan="", mem_block="", **kw):
    """ReAct with the model's native function-calling format (vLLM tool parser), stateful tool, forced final answer."""
    sys_msg = ("You are a careful problem solver. You may call the run_python tool for calculations. When you are "
               "finished, reply without a tool call and end with 'Final Answer: " + hint() + "'.")
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": f"{mem_block}{plan}Question: {ex['question']}"}]
    history = []
    for _ in range(MAX_STEPS):
        out = ctx.chat(llm, messages, tools=TOOLS, max_tokens=TASK["react_tok"])
        if not out.tool_calls:
            ans = extract_answer(out.text)
            if ans is not None:
                return ans
            messages.append({"role": "assistant", "content": out.text})
            break  # replied without a tool call or an answer -> ask for the final answer
        messages.append({"role": "assistant", "content": out.text or "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in out.tool_calls]})
        for cid, name, args in out.tool_calls:
            try:
                code = (json.loads(args) if isinstance(args, str) else args).get("code")
            except (json.JSONDecodeError, AttributeError):
                code = None
            if name != "run_python" or not code:
                ok, res = False, 'Error: call run_python with JSON arguments {"code": "<python source>"}'
            else:
                ok, res = run_repl(history, code)
            ctx.tool_calls += 1
            ctx.tool_ok += int(ok)
            messages.append({"role": "tool", "tool_call_id": cid, "content": res})
    ctx.info = {**ctx.info, "forced_final": True}
    messages.append({"role": "user", "content": "Stop using tools now. Give your final answer, ending with "
                                                "'Final Answer: " + hint() + "'."})
    out = ctx.chat(llm, messages, tools=TOOLS, tool_choice="none", max_tokens=TASK["react_tok"])
    return extract_answer(out.text)


def m_self_refine(llm, ctx, ex, **kw):
    g1 = ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"])
    ctx.cand = extract_answer(g1.text)
    crit = ctx.gen(llm, f"Question: {ex['question']}\nSolution:\n{g1.text}\n\nCritique the solution, finding any mistakes.", 400)
    g2 = ctx.gen(llm, f"Question: {ex['question']}\nSolution:\n{g1.text}\nCritique:\n{crit.text}\n\n"
                      f"Write an improved solution. End with 'Final Answer: {hint()}'.", TASK["cot_tok"])
    ans = extract_answer(g2.text)
    ctx.flagged = not same(ans, ctx.cand)
    return ans


def m_reflexion(llm, ctx, ex, memory=None, **kw):
    mb = memory.block(ex["question"]) if memory else ""
    g = ctx.gen(llm, cot_prompt(ex["question"], mb), TASK["cot_tok"])
    ctx.cand = extract_answer(g.text)
    return ctx.cand


def _pipeline(llm, ctx, ex, verifier, memory, use_plan=True, react_fn=None):
    mb = memory.block(ex["question"]) if memory else ""
    plan = ""
    if use_plan:
        plan = "Plan:\n" + ctx.gen(llm, f"Make a short numbered plan to solve this problem (no solution yet).\n\nQuestion: {ex['question']}\n", 150).text + "\n\n"
    ans = (react_fn or m_react)(llm, ctx, ex, plan=plan, mem_block=mb)
    ctx.cand = ans
    if verifier == "grounded":
        ok, val = verify_grounded(llm, ctx, ex["question"], ans)
        ctx.flagged = not ok
        if not ok and val is not None:
            ans = val  # trust the independent tool-derived value
    elif verifier == "intrinsic":
        ok = verify_intrinsic(llm, ctx, ex["question"], f"Final Answer: {ans if ans is None else show(ans)}")
        ctx.flagged = not ok
        if not ok:
            ans = m_cot(llm, ctx, ex)
    return ans


def m_always_on(llm, ctx, ex, verifier="grounded", memory=None, **kw):
    return _pipeline(llm, ctx, ex, verifier, memory)


def m_always_on_native(llm, ctx, ex, verifier="grounded", memory=None, **kw):
    return _pipeline(llm, ctx, ex, verifier, memory, react_fn=m_react_native)


def m_always_on_intrinsic(llm, ctx, ex, memory=None, **kw):
    return _pipeline(llm, ctx, ex, "intrinsic", memory)


def m_always_on_none(llm, ctx, ex, memory=None, **kw):
    return _pipeline(llm, ctx, ex, "none", memory)


def _gate(llm, ctx, ex, memory, agree_k=3, lp_thr=-0.35):
    """Cheap first stage: k sampled CoTs. Returns the agreed answer if unanimous and confident, else None."""
    if memory is not None and memory.lp_thr is not None:
        lp_thr = memory.lp_thr
    samples = [ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"], temperature=0.7) for _ in range(agree_k)]
    answers = [extract_answer(s.text) for s in samples]
    valid = [a for a in answers if a is not None]
    if valid:
        top, cnt = Counter(key(a) for a in valid).most_common(1)[0]
        conf = sum(s.mean_logprob for s in samples) / len(samples)
        ctx.info = {"agree": cnt, "conf": conf, "thr": lp_thr, "answers": [None if a is None else show(a) for a in answers]}
        if cnt >= agree_k and conf > lp_thr:
            ctx.cand = top
            return top
    ctx.escalated = True
    return None


def m_adaptive(llm, ctx, ex, verifier="grounded", memory=None, **kw):
    """Gate, then escalate disagreements / low confidence to the full agent pipeline."""
    top = _gate(llm, ctx, ex, memory)
    return top if top is not None else _pipeline(llm, ctx, ex, verifier, memory)


def m_sc5(llm, ctx, ex, k=5, **kw):
    """Self-consistency: majority vote over k sampled CoTs (compute-matched baseline)."""
    gens = [ctx.gen(llm, cot_prompt(ex["question"]), TASK["cot_tok"], temperature=0.7) for _ in range(k)]
    answers = [extract_answer(g.text) for g in gens]
    # Per-sample confidence and tokens let the adaptive gate (first 3 samples) be rebuilt offline from these runs.
    # raw values (floats for numeric tasks, normalized strings for MATH): show() rounds floats to 6 significant digits
    ctx.info = {"answers": answers, "lps": [g.mean_logprob for g in gens],
                "n_in": [g.n_in for g in gens], "n_out": [g.n_out for g in gens]}
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    best = Counter(key(a) for a in valid).most_common(1)[0][0]
    return next(a for a in valid if key(a) == best)


def m_think(llm, ctx, ex, **kw):
    """Native reasoning (Qwen3 thinking mode). Answer is read only after </think>; truncated thinking -> no answer."""
    g = ctx.gen(llm, cot_prompt(ex["question"]), TASK["think_tok"], think=True)
    done = "</think>" in g.text
    ctx.info = {**ctx.info, "think_truncated": not done, "think_tokens": g.n_out}
    return extract_answer(g.text.split("</think>")[-1]) if done else None


def m_gate_think(llm, ctx, ex, memory=None, **kw):
    """Gate, then escalate disagreements / low confidence to thinking mode instead of the tool pipeline."""
    top = _gate(llm, ctx, ex, memory)
    return top if top is not None else m_think(llm, ctx, ex)


METHODS = {
    "direct": m_direct, "cot": m_cot, "react": m_react, "self_refine": m_self_refine,
    "reflexion": m_reflexion, "always_on": m_always_on, "adaptive": m_adaptive,
    "always_on_intrinsic": m_always_on_intrinsic, "always_on_none": m_always_on_none,
    "sc5": m_sc5, "think": m_think, "gate_think": m_gate_think,
    "react_repl": m_react_repl, "react_native": m_react_native, "always_on_native": m_always_on_native,
}
NEEDS_MEMORY = {"reflexion", "always_on", "adaptive", "always_on_intrinsic", "always_on_none", "gate_think",
                "always_on_native"}
NATIVE_METHODS = {"react_native", "always_on_native"}  # need the vLLM server backend (native tool parsing)
THINK_METHODS = {"think", "gate_think"}
