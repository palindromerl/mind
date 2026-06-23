# mind

**A tiny, from-scratch transformer you can grow incrementally — in ~300 lines of pure numpy.**

No PyTorch, no TensorFlow, no JAX, no Hugging Face. Just `numpy`. Every matmul, every gradient, every attention head is hand-written. You start with an empty model, feed it a book, ask it questions, feed it another book, and watch it accumulate knowledge over time.

`mind.py` is one file. The model state lives in `model.pkl`. Each `train` call loads what you've taught it so far, learns a bit more, and saves it back — so the same checkpoint gets smarter every time you run it.

---

## Architecture

A small decoder-only transformer, built by hand:

```
tokens ──► embedding (M)
              + position embedding (P)
              ├──► [LayerNorm → causal self-attention → residual] × L
              └──► final LayerNorm
                        └──► output projection (Uo) → softmax over vocab
```

Defaults (override-able in source):

| Hyperparameter | Default | Meaning                                  |
|----------------|---------|------------------------------------------|
| `D`            | `96`    | embedding width                          |
| `T`            | `32`    | context length (tokens)                  |
| `L`            | `2`     | number of self-attention layers          |
| `MIN_COUNT`    | `3`     | times a word must appear to join vocab   |
| `REPLAY_CAP`   | `150,000` | tokens of past text kept against forgetting |

Optimizer: hand-rolled Adam with a cosine LR schedule and 500-step warmup. Loss: standard next-token cross-entropy (with a label mask so QA training only scores the *answer* tokens, not the question).

Tokenization is **word-level**: lowercase, alpha-only (`re.sub(r"[^a-z\s]", " ", text.lower()).split()`). The original casing of every surface form is remembered separately in `m["surface"]` so the `--grammar` flag can restore proper nouns on output.

Four reserved special tokens drive the instruction-tuning protocol: `<unk>`, `question`, `answer`, `end`.

---

## Quickstart

```bash
pip install numpy
# optional, only for `genqa`:
pip install anthropic
```

```bash
# 1) Teach it a book (pretraining-style next-token prediction)
python mind.py train book.txt

# 2) Optionally use an LLM to mine Q/A pairs from that book
python mind.py genqa book.txt -o qa.txt

# 3) Instruction-tune on those pairs (loss only on the answer span)
python mind.py train qa.txt --qa

# 4) Talk to it
python mind.py ask "what sport does john play" --grammar
python mind.py gen "john thornton" -n 30 --temp 0.9 --grammar
```

State persists in `model.pkl` by default; pass `--model my.pkl` to use a different checkpoint.

---

## CLI

All commands accept `--model PATH` (default `model.pkl`).

### `train FILE [--qa] [--steps N]`
Loads the checkpoint (or creates an empty one), ingests `FILE`, trains, and saves back.

- **Without `--qa`** → free-text pretraining. Adds new words to the vocab once they've appeared `MIN_COUNT` times across all data seen so far. 70% of training batches sample the new text, 30% sample the replay buffer, to fight catastrophic forgetting.
- **With `--qa`** → instruction tuning on tab-separated `question<TAB>answer` lines (one per line; blank lines and lines without a tab are skipped). Every word in the Q/A file joins the vocab immediately (curated data is treated as signal, not noise). The loss mask zeroes out the question tokens — gradients only flow through the answer and the closing `end` token. 28% of batches still sample from the replay buffer.
- `--steps N` overrides the auto-chosen step count. Defaults are `max(3000, min(20000, 4·tokens/T))` for free text and `max(8000, 40·pairs)` for QA.

### `genqa FILE [-o OUT] [--llm MODEL] [--chunk N]`
A *separate* process that uses an LLM to **manufacture** training Q/A pairs from a passage of text. It chunks the text into `--chunk` (default 1200) word windows, asks the LLM to emit short, fully-grounded `question<TAB>answer` lines, de-duplicates them, and writes them to `OUT` (default `FILE.qa.txt`).

- Provider: **Anthropic**. Requires the `anthropic` Python package and the `ANTHROPIC_API_KEY` environment variable.
- Default model: `claude-sonnet-4-6` (override with `--llm`).
- The LLM is asked for several paraphrased forms of each question and short (1–6 word) answers — the format `mind.py` trains on best.

This is the *only* place an external LLM is used. Training, generation, and answering are 100% in-repo numpy.

### `ask "QUESTION" [--temp T] [--grammar]`
Wraps your prompt as `question … answer …` and decodes up to 14 tokens (stops on `end`). `--temp 0` is greedy; otherwise samples from the top-5 logits at temperature `T`. `--grammar` re-applies learned casing and adds a terminal period.

### `gen "SEED" [-n N] [--temp T] [--grammar]`
Free-form continuation. Encodes the seed, decodes up to `N` tokens (default 30) at temperature `--temp` (default 0.9). `--grammar` truecases and punctuates.

---

## How it grows

Three mechanisms let the same checkpoint absorb new material without erasing old material:

1. **Vocab growth.** `extend_vocab` appends new rows to the token embedding `M` and new columns to the output projection `Uo`, initialized small. Existing rows/columns — and therefore everything the model already knows about previously-seen words — are preserved exactly.
2. **Replay buffer.** Every pretraining run appends its token stream to `m["replay"]` and clips to the last `REPLAY_CAP` tokens. During training, 30% of batches (free-text) or 28% (QA) are drawn from replay, so old distributions keep showing up in gradients.
3. **Frequency-gated vocab.** Free-text training only promotes a word into the vocab after it's been seen `MIN_COUNT` times across all runs. Curated QA data bypasses this gate — every QA word is added immediately.

Surface forms (original casing of every token) accumulate in `m["surface"]` so `prettify` can do rule-based truecasing without any model call: known proper nouns are recased, `i` becomes `I`, the first character is capitalized, and a terminal `.` is appended if one isn't there.

`model.pkl` is a single pickle of the whole dict — params, vocab, counts, surface map, replay buffer, and a `tuned` flag. Move it between machines, branch it, snapshot it.

---

## Generating Q/A pairs from a book

The intended workflow:

```bash
# pretrain on the raw book so the words exist in the vocab and the LM has context
python mind.py train call_of_the_wild.txt

# use Claude to mine factual Q/A pairs grounded in the book
export ANTHROPIC_API_KEY=sk-ant-...
python mind.py genqa call_of_the_wild.txt -o cotw_qa.txt --llm claude-sonnet-4-6

# instruction-tune on those pairs
python mind.py train cotw_qa.txt --qa

# ask it things
python mind.py ask "who buys buck" --grammar
python mind.py ask "what sled does john thornton drive" --grammar
```

The system prompt inside `genqa` asks for **fully answerable** questions, **short (1–6 word) answers**, and **multiple paraphrases** per question — the format the tiny model actually has the capacity to learn at `D=96, L=2`.

---

## Limitations (read this part)

`mind.py` is a teaching artifact and a personal-scale experiment, not a production language model:

- **Toy scale.** `D=96`, `L=2`, `T=32`. It will not write essays. It will sometimes answer short factual questions about material it has been drilled on.
- **Word-level tokenization.** No BPE, no subwords. Out-of-vocab words become `<unk>`.
- **CPU-only numpy.** Training a novel-sized corpus takes minutes, not seconds. No batching across sequences — one sequence per step.
- **No KV cache.** Generation re-runs the full forward pass each step (the context is always re-fed). Fine at `T=32`; would be silly at `T=2048`.
- **Replay is a sliding window, not a real continual-learning algorithm.** Long-term, the oldest tokens still fall off the back of `REPLAY_CAP`.
- **`genqa` calls an external LLM.** Nothing else in this repo does.

The goal of this code is to be **legible**: every line of attention, every gradient, the whole optimizer, the whole tokenizer, the whole training loop — all visible, all editable, all under 350 lines.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

```
@software{mind2026,
  author = {Palindrome Research Labs},
  title  = {mind: a tiny from-scratch numpy transformer},
  year   = {2026},
  url    = {https://github.com/palindromerl/mind}
}
```
