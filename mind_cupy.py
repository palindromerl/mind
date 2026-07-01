#!/usr/bin/env python3
"""
mind_cupy.py - CuPy-accelerated version of mind.py

  python mind_cupy.py train  book.txt              # add a book to pretraining
  python mind_cupy.py train  qa.txt --qa           # add Q/A pairs (instruction tuning)
  python mind_cupy.py genqa  book.txt -o qa.txt    # use an LLM to MAKE Q/A pairs from a book
  python mind_cupy.py ask    "what sport does john play"  [--temp 0.7] [--grammar]
  python mind_cupy.py gen    "john thornton"       [--temp 0.9] [--grammar] [-n 30]
  python mind_cupy.py train  book.txt --no-gpu     # force CPU mode

State lives in model.pkl (override with --model). Every train call loads that
state, trains more, and saves it back -- so knowledge accumulates across runs.
No ML libraries: just numpy/cupy. Same architecture we built by hand
(token + position embeddings -> causal self-attention x L -> LayerNorm -> softmax).

GPU ACCELERATION:
  Model params are stored as numpy arrays in pickle (for compatibility).
  At the start of fb()/fwd_logits(), they are transferred to CuPy arrays.
  Gradients are converted back to numpy before the Adam step.
  Use --gpu (default) or --no-gpu to switch at runtime.
"""
import argparse, pickle, os, re, sys, math, random, time
import numpy as np
from collections import Counter

# ----------------------------- GPU setup -----------------------------------
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    cp = None
    _cupy_available = False

USE_GPU = True  # runtime flag; overridden by --gpu / --no-gpu

def xp():
    """Return cupy or numpy depending on USE_GPU flag."""
    if USE_GPU and _cupy_available:
        return cp
    return np

def to_gpu(arr):
    """Move a numpy array to GPU if USE_GPU is set."""
    if USE_GPU and _cupy_available:
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move a cupy array back to numpy (no-op if already numpy)."""
    if USE_GPU and _cupy_available and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr

def params_to_gpu(p):
    """Convert all params dict values to GPU arrays."""
    return {k: to_gpu(v) for k, v in p.items()}

def grads_to_cpu(g):
    """Convert all grad dict values back to numpy."""
    return {k: to_cpu(v) for k, v in g.items()}

def sync():
    """Synchronize GPU for accurate timing."""
    if USE_GPU and _cupy_available:
        cp.cuda.Stream.null.synchronize()

# ----------------------------- config defaults -----------------------------
D, T, L = 96, 32, 2          # width, context length, number of attention layers
MIN_COUNT = 3                # a new word joins the vocab once seen this many times
REPLAY_CAP = 150_000         # tokens of past text kept to fight forgetting
SPECIALS = ["<unk>", "question", "answer", "end"]

# ----------------------------- tokenizer -----------------------------------
def tokenize(text):
    return re.sub(r"[^a-z\s]", " ", text.lower()).split()

def raw_tokens(text):                      # keep original case, for pretty-printing later
    return re.sub(r"[^A-Za-z\s]", " ", text).split()

# ----------------------------- model state ---------------------------------
def rinit(*s): return (np.random.randn(*s) * 0.05)

def new_model():
    m = {"cfg": {"D": D, "T": T, "L": L},
         "vocab": list(SPECIALS), "counts": Counter(), "surface": {},
         "replay": [], "tuned": False, "params": {}}
    V = len(m["vocab"]); cfg = m["cfg"]; d = cfg["D"]
    p = m["params"]
    # Keep as numpy for pickle compatibility
    p["M"] = rinit(V, d); p["P"] = rinit(cfg["T"], d); p["Uo"] = rinit(d, V)
    for l in range(cfg["L"]):
        p[f"Wq{l}"] = rinit(d, d); p[f"Wk{l}"] = rinit(d, d); p[f"Wv{l}"] = rinit(d, d)
        p[f"g{l}"] = np.ones(d); p[f"b{l}"] = np.zeros(d)
    p["gf"] = np.ones(d); p["bf"] = np.zeros(d)
    return m

def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f: return pickle.load(f)
    return new_model()

def save_model(m, path):
    # Ensure all params are numpy before pickling
    m["params"] = {k: to_cpu(v) for k, v in m["params"].items()}
    with open(path, "wb") as f: pickle.dump(m, f)

def w2i(m): return {w: i for i, w in enumerate(m["vocab"])}

def extend_vocab(m, new_words):
    if not new_words: return
    cfg = m["cfg"]; d = cfg["D"]; p = m["params"]
    for w in new_words: m["vocab"].append(w)
    n = len(new_words)
    p["M"] = np.vstack([p["M"], rinit(n, d)])
    p["Uo"] = np.hstack([p["Uo"], rinit(d, n)])

# ----------------------------- forward / backward --------------------------
def smax(z):
    _xp = xp()
    z = z - z.max(1, keepdims=True)
    e = _xp.exp(z)
    return e / e.sum(1, keepdims=True)

def ln_f(x, g, b):
    _xp = xp()
    mu = x.mean(1, keepdims=True); xc = x - mu
    inv = 1 / _xp.sqrt((xc**2).mean(1, keepdims=True) + 1e-5); xh = xc * inv
    return xh * g + b, (xh, inv, g)

def ln_b(dy, c):
    _xp = xp()
    xh, inv, g = c
    dg = (dy*xh).sum(0); db = dy.sum(0); dxh = dy*g
    dx = inv*(dxh - dxh.mean(1, keepdims=True) - xh*(dxh*xh).mean(1, keepdims=True))
    return dx, dg, db

def fb(m, idx, lmask=None):
    """forward+backward on one sequence; returns loss and grads. lmask: which targets count."""
    _xp = xp()
    cfg = m["cfg"]; Ln = len(idx); scale = 1/_xp.sqrt(_xp.array(cfg["D"], dtype=_xp.float64))

    # Transfer params to GPU for this forward/backward pass
    p = params_to_gpu(m["params"])

    msk = _xp.triu(_xp.ones((Ln, Ln)), 1).astype(bool)
    idx_gpu = _xp.array(idx)
    h = p["M"][idx_gpu] + p["P"][:Ln]
    lnc = []; atc = []

    for l in range(cfg["L"]):
        n, c1 = ln_f(h, p[f"g{l}"], p[f"b{l}"])
        Q = n@p[f"Wq{l}"]; K = n@p[f"Wk{l}"]; Vv = n@p[f"Wv{l}"]
        S = _xp.where(msk, -1e9, (Q@K.T)*scale); A = smax(S); h = h + A@Vv
        lnc.append(c1); atc.append((n, Q, K, Vv, A))

    hf, lf = ln_f(h, p["gf"], p["bf"]); logits = hf@p["Uo"]
    tgt = _xp.array(idx[1:]); pos = _xp.arange(Ln-1)
    z = logits - logits.max(1, keepdims=True); E = _xp.exp(z); Ps = E/E.sum(1, keepdims=True)
    lm = _xp.ones(Ln-1, dtype=bool) if lmask is None else _xp.array(lmask)
    ls = -_xp.log(Ps[pos, tgt] + 1e-9)
    loss = float(ls[lm].mean()) if lm.any() else 0.0

    dl = _xp.zeros_like(logits)
    P_ = Ps[:Ln-1].copy()
    P_[pos, tgt] -= 1
    lm_sum = int(lm.sum())
    dl[:Ln-1] = _xp.where(lm[:, None], P_/max(lm_sum, 1), 0.0)

    g = {}
    g["Uo"] = hf.T@dl; dhf = dl@p["Uo"].T
    dh, g["gf"], g["bf"] = ln_b(dhf, lf)

    for l in reversed(range(cfg["L"])):
        n, Q, K, Vv, A = atc[l]; dZ = dh.copy()
        dA = dZ@Vv.T; dVv = A.T@dZ; tmp = (dA*A).sum(1, keepdims=True)
        dS = A*(dA - tmp); dS[msk] = 0
        dQ = (dS@K)*scale; dK = (dS.T@Q)*scale
        g[f"Wq{l}"] = n.T@dQ; g[f"Wk{l}"] = n.T@dK; g[f"Wv{l}"] = n.T@dVv
        dn = dQ@p[f"Wq{l}"].T + dK@p[f"Wk{l}"].T + dVv@p[f"Wv{l}"].T
        dx, dg_l, db = ln_b(dn, lnc[l]); dh = dh + dx; g[f"g{l}"] = dg_l; g[f"b{l}"] = db

    dM = _xp.zeros_like(p["M"])
    _xp.add.at(dM, idx_gpu, dh)
    g["M"] = dM; g["P"] = _xp.zeros_like(p["P"]); g["P"][:Ln] = dh

    # Convert gradients back to numpy for Adam step
    return loss, grads_to_cpu(g)

class Adam:
    def __init__(self, params):
        # Adam state stays on CPU (numpy)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            # Both params and grads are numpy here
            g = grads[k]
            self.m[k] = 0.9*self.m[k] + 0.1*g
            self.v[k] = 0.999*self.v[k] + 0.001*g*g
            mh = self.m[k]/(1-0.9**self.t)
            vh = self.v[k]/(1-0.999**self.t)
            params[k] -= lr*mh/(np.sqrt(vh)+1e-8)

def lr_sched(s, total, base, warm=500):
    return base*s/warm if s < warm else 0.5*base*(1+math.cos(math.pi*(s-warm)/max(1, total-warm)))

# ----------------------------- training commands ---------------------------
def train_text(m, path, steps, base_lr=0.005):
    text = open(path, encoding="utf-8", errors="ignore").read()
    # learn surface casing for pretty-printing
    for w in raw_tokens(text):
        d = m["surface"].setdefault(w.lower(), Counter()); d[w] += 1
    toks = tokenize(text); m["counts"].update(toks)
    cur = w2i(m)
    new = sorted({w for w in set(toks) if w not in cur and m["counts"][w] >= MIN_COUNT})
    extend_vocab(m, new); cur = w2i(m)
    unk = cur["<unk>"]; stream = [cur.get(w, unk) for w in toks]
    replay = m["replay"]
    mode_str = f"GPU (CuPy {cp.__version__})" if (USE_GPU and _cupy_available) else "CPU (NumPy)"
    print(f"  ingested {len(toks):,} tokens | +{len(new)} new words | vocab now {len(m['vocab']):,}")
    print(f"  compute mode: {mode_str}")
    steps = steps or max(3000, min(20000, 4*len(stream)//m['cfg']['T']))
    opt = Adam(m["params"]); T_ = m["cfg"]["T"]; run = 0.0
    pool = stream + replay
    report_interval = max(1, steps//5)
    # Timing
    t0 = time.perf_counter(); step_t0 = t0; tokens_in_interval = 0
    for s in range(steps):
        src = stream if (not replay or random.random() < 0.7) else replay
        if len(src) <= T_+1: src = pool
        i = random.randint(0, len(src)-T_-1)
        loss, g = fb(m, src[i:i+T_])
        opt.step(m["params"], g, lr_sched(s, steps, base_lr))
        run += loss
        tokens_in_interval += T_
        if (s+1) % report_interval == 0:
            sync()
            elapsed = time.perf_counter() - step_t0
            tok_s = tokens_in_interval / elapsed if elapsed > 0 else 0
            print(f"  step {s+1}/{steps}  loss {run/report_interval:.3f}  {tok_s:.0f} tok/s")
            run = 0.0; step_t0 = time.perf_counter(); tokens_in_interval = 0

    sync()
    total_elapsed = time.perf_counter() - t0
    total_tokens = steps * T_
    overall_tok_s = total_tokens / total_elapsed
    print(f"  --- Training summary ---")
    print(f"  Total time: {total_elapsed:.2f}s for {steps} steps ({total_tokens:,} tokens)")
    print(f"  Overall throughput: {overall_tok_s:.0f} tok/s")
    # update replay buffer (keep a capped sample of everything seen)
    m["replay"] = (replay + stream)[-REPLAY_CAP:]
    return m

def parse_qa(path):
    pairs = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        if not line.strip() or "\t" not in line: continue
        q, a = line.split("\t", 1); pairs.append((q.strip().lower(), a.strip().lower()))
    return pairs

def train_qa(m, path, steps, base_lr=0.002):
    pairs = parse_qa(path)
    cur = w2i(m); need = set()
    for q, a in pairs:
        for w in tokenize(q)+tokenize(a):
            if w not in cur: need.add(w)
    extend_vocab(m, sorted(need)); cur = w2i(m)
    enc = lambda s: [cur[w] for w in tokenize(s) if w in cur]
    examples = []
    for q, a in pairs:
        seq = [cur["question"]]+enc(q)+[cur["answer"]]+enc(a)+[cur["end"]]
        if len(seq) > m["cfg"]["T"] or len(enc(a)) == 0: continue
        lm = np.zeros(len(seq)-1, bool); lm[len(enc(q))+1:] = True
        examples.append((seq, lm))
    mode_str = f"GPU (CuPy {cp.__version__})" if (USE_GPU and _cupy_available) else "CPU (NumPy)"
    print(f"  {len(examples)} usable QA pairs | +{len(need)} new words | vocab {len(m['vocab']):,}")
    print(f"  compute mode: {mode_str}")
    steps = steps or max(8000, 40*len(examples))
    opt = Adam(m["params"]); T_ = m["cfg"]["T"]; replay = m["replay"]; run = 0.0
    report_interval = max(1, steps//5)
    t0 = time.perf_counter(); step_t0 = t0; tokens_in_interval = 0
    for s in range(steps):
        if random.random() < 0.72 or not replay:
            seq, lm = random.choice(examples); loss, g = fb(m, seq, lm)
            tokens_in_interval += len(seq)
        else:
            i = random.randint(0, len(replay)-T_-1); loss, g = fb(m, replay[i:i+T_])
            tokens_in_interval += T_
        opt.step(m["params"], g, lr_sched(s, steps, base_lr)); run += loss
        if (s+1) % report_interval == 0:
            sync()
            elapsed = time.perf_counter() - step_t0
            tok_s = tokens_in_interval / elapsed if elapsed > 0 else 0
            print(f"  step {s+1}/{steps}  loss {run/report_interval:.3f}  {tok_s:.0f} tok/s")
            run = 0.0; step_t0 = time.perf_counter(); tokens_in_interval = 0

    sync()
    total_elapsed = time.perf_counter() - t0
    print(f"  --- Training summary ---")
    print(f"  Total time: {total_elapsed:.2f}s for {steps} steps")
    m["tuned"] = True
    return m

# ----------------------------- generation ----------------------------------
def fwd_logits(m, idx):
    _xp = xp()
    cfg = m["cfg"]; Ln = len(idx); scale = 1/_xp.sqrt(_xp.array(cfg["D"], dtype=_xp.float64))

    # Transfer params to GPU for inference
    p = params_to_gpu(m["params"])

    msk = _xp.triu(_xp.ones((Ln, Ln)), 1).astype(bool)
    idx_gpu = _xp.array(idx)
    h = p["M"][idx_gpu] + p["P"][:Ln]

    for l in range(cfg["L"]):
        n, _ = ln_f(h, p[f"g{l}"], p[f"b{l}"])
        Q = n@p[f"Wq{l}"]; K = n@p[f"Wk{l}"]; Vv = n@p[f"Wv{l}"]
        S = _xp.where(msk, -1e9, (Q@K.T)*scale); h = h + smax(S)@Vv
    hf, _ = ln_f(h, p["gf"], p["bf"])
    result = (hf@p["Uo"])[-1]
    return to_cpu(result)

def sample(m, ids, n, temp, topk=5):
    cur = w2i(m); end = cur["end"]; T_ = m["cfg"]["T"]; out = []
    for _ in range(n):
        logits = fwd_logits(m, ids[-T_:])
        if temp <= 0:
            nx = int(np.argmax(logits))
        else:
            logits = logits/temp; top = np.argsort(-logits)[:topk]
            pr = np.exp(logits[top]-logits[top].max()); pr /= pr.sum()
            nx = int(np.random.choice(top, p=pr))
        if nx == end: break
        out.append(nx); ids.append(nx)
    return out

def ask(m, q, temp, grammar):
    cur = w2i(m)
    ids = [cur["question"]] + [cur[w] for w in tokenize(q) if w in cur] + [cur["answer"]]
    out = sample(m, ids, n=14, temp=temp)
    words = [m["vocab"][i] for i in out]
    return prettify(m, words) if grammar else " ".join(words)

def gen(m, seed, n, temp, grammar):
    cur = w2i(m); ids = [cur[w] for w in tokenize(seed) if w in cur] or [cur["<unk>"]]
    out = sample(m, ids, n=n, temp=temp)
    words = tokenize(seed) + [m["vocab"][i] for i in out]
    return prettify(m, words) if grammar else " ".join(words)

# ----------------------------- grammar (non-LLM) ---------------------------
def surface_of(m, t):
    if t == "i": return "I"
    c = m["surface"].get(t)
    if c:
        best = c.most_common(1)[0][0]
        return best if best[:1].isupper() else t
    return t

def prettify(m, words):
    """Rule-based truecasing + terminal punctuation."""
    if not words: return ""
    s = " ".join(surface_of(m, w) for w in words)
    s = re.sub(r"\bi\b", "I", s)
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?": s += "."
    return s

# ----------------------------- genqa (LLM) ---------------------------------
def genqa(path, out, model_name, chunk_words):
    """Use an LLM to extract answerable Q/A pairs from a text."""
    try:
        import anthropic
    except ImportError:
        sys.exit("genqa needs the anthropic package:  pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("genqa needs ANTHROPIC_API_KEY set in your environment.")
    client = anthropic.Anthropic()
    words = open(path, encoding="utf-8", errors="ignore").read().split()
    chunks = [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]
    sys_prompt = ("You extract training questions from text. For the passage, output factual "
                  "question-and-answer pairs that are fully answerable from the passage alone. "
                  "Keep answers SHORT (1-6 words). Write several paraphrased versions of each "
                  "question. Output ONLY lines of the form:  question<TAB>answer  with a real tab "
                  "character between them. No numbering, no preamble, no blank explanations.")
    seen = set(); n = 0
    with open(out, "w", encoding="utf-8") as f:
        for ci, ch in enumerate(chunks):
            print(f"  chunk {ci+1}/{len(chunks)} -> LLM ...")
            msg = client.messages.create(model=model_name, max_tokens=2000,
                    system=sys_prompt, messages=[{"role": "user", "content": ch}])
            for line in "".join(b.text for b in msg.content if b.type == "text").splitlines():
                if "\t" in line:
                    key = line.strip().lower()
                    if key not in seen:
                        seen.add(key); f.write(line.strip()+"\n"); n += 1
    print(f"  wrote {n} pairs -> {out}")

# ----------------------------- CLI -----------------------------------------
def main():
    global USE_GPU

    ap = argparse.ArgumentParser(description="grow a tiny transformer (CuPy GPU-accelerated)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=True,
                    help="Use GPU via CuPy (default)")
    ap.add_argument("--no-gpu", dest="gpu", action="store_false",
                    help="Force CPU (NumPy) mode")

    pt = sub.add_parser("train"); pt.add_argument("file"); pt.add_argument("--qa", action="store_true")
    pt.add_argument("--steps", type=int, default=0)
    pt.add_argument("--D", type=int, default=0, help="Embedding dimension (overrides default)")
    pt.add_argument("--T", type=int, default=0, help="Context length (overrides default)")
    pt.add_argument("--L", type=int, default=0, help="Number of attention layers (overrides default)")
    pt.add_argument("--batch-size", type=int, default=64, help="Batch size (informational)")
    pg = sub.add_parser("genqa"); pg.add_argument("file"); pg.add_argument("-o", "--out", default=None)
    pg.add_argument("--llm", default="claude-sonnet-4-6"); pg.add_argument("--chunk", type=int, default=1200)
    pa = sub.add_parser("ask"); pa.add_argument("q"); pa.add_argument("--temp", type=float, default=0.0)
    pa.add_argument("--grammar", action="store_true")
    pn = sub.add_parser("gen"); pn.add_argument("seed"); pn.add_argument("-n", type=int, default=30)
    pn.add_argument("--temp", type=float, default=0.9); pn.add_argument("--grammar", action="store_true")
    a = ap.parse_args()

    USE_GPU = a.gpu
    if USE_GPU and not _cupy_available:
        print("WARNING: CuPy not available, falling back to NumPy CPU mode.", file=sys.stderr)
        USE_GPU = False

    if USE_GPU and _cupy_available:
        try:
            cp.cuda.Device(0).use()
            dev_count = cp.cuda.runtime.getDeviceCount()
            dev_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = dev_props['name'].decode() if isinstance(dev_props['name'], bytes) else dev_props['name']
            print(f"  GPU: {gpu_name} ({dev_count} device(s)) | CuPy {cp.__version__}")
        except Exception as e:
            print(f"WARNING: GPU init failed ({e}), falling back to CPU.", file=sys.stderr)
            USE_GPU = False

    if a.cmd == "genqa":
        genqa(a.file, a.out or (a.file + ".qa.txt"), a.llm, a.chunk); return
    if a.cmd == "train":
        # Override global D, T, L if specified on command line
        global D, T, L
        if hasattr(a, 'D') and a.D > 0:
            D = a.D
        if hasattr(a, 'T') and a.T > 0:
            T = a.T
        if hasattr(a, 'L') and a.L > 0:
            L = a.L
        print(f"  Model config: D={D}, T={T}, L={L}")
        m = load_model(a.model)
        # If loading a new model, rebuild it with the updated D/T/L
        if not m["params"]:
            m = new_model()
        m = train_qa(m, a.file, a.steps) if a.qa else train_text(m, a.file, a.steps)
        save_model(m, a.model); print(f"  saved -> {a.model}"); return
    m = load_model(a.model)
    if not m["params"]: sys.exit("model is empty; train a book first.")
    if a.cmd == "ask": print(ask(m, a.q, a.temp, a.grammar))
    if a.cmd == "gen": print(gen(m, a.seed, a.n, a.temp, a.grammar))

if __name__ == "__main__":
    np.random.seed(0); random.seed(0); main()
