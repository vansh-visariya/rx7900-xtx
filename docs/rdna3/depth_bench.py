#!/usr/bin/env python3
"""
Long-context benchmark for an already running llama-server (agentic-coding shaped).

For each requested depth, builds a chat prompt of about that many tokens from real source
code (llama.cpp sources), with a unique first line so the prompt cache cannot reuse an
earlier prefix, asks a coding question with thinking on, and reports:
  prompt tokens, cold prefill t/s, generated tokens, decode t/s, draft acceptance.

  depth_bench.py --port 8080 --depths 16000,64000,128000 --n-predict 1024

The prompts are read with git from this repository at --corpus-ref (needs a full clone).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SRC_GLOBS = [
    "src/*.cpp",
    "src/models/*.cpp",
    "common/*.cpp",
    "tools/server/*.cpp",
    "ggml/src/ggml-vulkan/*.cpp",
    "ggml/src/ggml-cpu/*.c",
    "ggml/src/*.c",
]

TASKS = {
    "explain": ("Task: find the function in these files that handles speculative draft acceptance, explain step by "
                "step how it decides which drafted tokens to keep, then write an improved version of it with comments."),
    # copy-heavy, like an agent's file edit: most of the answer repeats code from the context
    "edit": ("Task: output the first 120 lines of the file src/llama-adapter.cpp shown above, unchanged except that "
             "you add a one-line comment above each function saying what it does. Output only the code."),
}


def post(port, path, payload, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def corpus(ref):
    # sources as of a fixed commit, so edits to the working tree do not change the prompts
    import fnmatch
    import subprocess
    repo = REPO
    names = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref],
                           capture_output=True, text=True, check=True).stdout.split()
    parts = []
    for g in SRC_GLOBS:
        pat = g
        for f in sorted(n for n in names if fnmatch.fnmatch(n, pat) and os.path.dirname(n) == os.path.dirname(pat)):
            txt = subprocess.run(["git", "-C", repo, "show", f"{ref}:{f}"], capture_output=True, text=True,
                                 errors="replace", check=True).stdout
            parts.append(f"// ===== {f} =====\n" + txt)
    return "\n".join(parts)


def build_prompt(port, code, n_tok, nonce, task):
    # code is ~3.1 chars/token; refine once with the tokenizer
    chars = int(n_tok * 3.1)
    for _ in range(3):
        msg = f"Session {nonce}.\n\nHere are source files from the project:\n\n{code[:chars]}\n\n{task}"
        rendered = post(port, "/apply-template", {"messages": [{"role": "user", "content": msg}]})["prompt"]
        n = len(post(port, "/tokenize", {"content": rendered})["tokens"])
        if abs(n - n_tok) < 0.02 * n_tok:
            break
        chars = int(chars * n_tok / n)
    return rendered, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--depths", default="16000,64000,128000")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--task", choices=sorted(TASKS), default="explain")
    ap.add_argument("--out", default="")
    ap.add_argument("--corpus-ref", default="89be5cc8a", help="commit of this repository the source corpus is read from")
    ap.add_argument("--random-nonce", action="store_true",
                    help="random first line per request (default: fixed per depth; the server restarts per run)")
    a = ap.parse_args()

    code = corpus(a.corpus_ref)
    rows = []
    for d in [int(x) for x in a.depths.split(",")]:
        nonce = f"{random.randrange(1 << 30):08x}" if a.random_nonce else f"d{d}s{a.seed}"
        prompt, n = build_prompt(a.port, code, d, nonce, TASKS[a.task])
        t0 = time.time()
        r = post(a.port, "/completion", {
            "prompt": prompt, "n_predict": a.n_predict, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "seed": a.seed, "cache_prompt": True,
        })
        wall = time.time() - t0
        t = r.get("timings", {})
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        row = {
            "depth": n, "prompt_n": t.get("prompt_n"), "prefill_t_s": t.get("prompt_per_second"),
            "gen_n": t.get("predicted_n"), "decode_t_s": t.get("predicted_per_second"),
            "acc": (da / dn) if dn else None, "wall_s": wall,
            # equal text hash and draft counts between two builds = bit-exact end to end (same seed)
            "draft_n": dn, "draft_n_accepted": da,
            "text_hash": hashlib.sha256(r.get("content", "").encode()).hexdigest()[:16],
            "content": r.get("content", ""),
        }
        rows.append(row)
        print(f"depth {n:7d}: prefill {row['prompt_n']} tok @ {row['prefill_t_s']:.1f} t/s | "
              f"decode {row['gen_n']} tok @ {row['decode_t_s']:.2f} t/s | acc {row['acc'] if row['acc'] is None else round(row['acc'], 3)} | wall {wall:.0f}s",
              flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
