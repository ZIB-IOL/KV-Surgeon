"""Build data/lca_repo_completion.json from Long Code Arena project-level code completion.

Source: JetBrains-Research/lca-project-level-code-completion, config large_context
(Bogomolov et al. 2024, arXiv:2406.11612). Keeps the first datapoint of each repository
whose .py files plus completion file fit --max-gemma-tokens, until --n-repos repositories.
The context is the repository's .py files joined with `[start of <path>]` markers. Each
completion line becomes one question (file prefix, write the next line), scored by exact
match of the first generated line; the line class is stored as task `lca_<class>`.

    python scripts/build_lca_repo_completion.py
"""
import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path(__file__).parent.parent / "data" / "lca_repo_completion.json"
HF_NAME = "JetBrains-Research/lca-project-level-code-completion"
HF_CONFIG = "large_context"
CLASSES = ("infile", "inproject", "commited", "common", "non_informative", "random")

INSTRUCTION = (
    "The repository above is the project context. Below is the beginning of the "
    "file `{path}` from the same repository, cut off right before one line. "
    "Write that next line of the file.\n\n```python\n{prefix}\n```\n\n"
    "Respond with only the next line of code, nothing else."
)


def build_context(snapshot) -> str:
    parts = []
    for fname, content in zip(snapshot["filename"], snapshot["content"]):
        if not fname.endswith(".py") or not content:
            continue
        parts.append(f"[start of {fname}]\n{content.rstrip()}\n")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-repos", type=int, default=20)
    ap.add_argument("--max-gemma-tokens", type=int, default=120000,
                    help="cap on gemma tokens of (py context + completion file)")
    ap.add_argument("--max-prefix-chars", type=int, default=16000,
                    help="keep only the last N characters of the file prefix in each question")
    ap.add_argument("--gemma-tokenizer", default="google/gemma-3-12b-it")
    ap.add_argument("--qwen-tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    gem = AutoTokenizer.from_pretrained(args.gemma_tokenizer)
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer)
    for t in (gem, qwen):
        t.model_max_length = int(1e9)

    def ng(s):
        return len(gem.encode(s, add_special_tokens=False))

    def nq(s):
        return len(qwen.encode(s, add_special_tokens=False))

    ds = load_dataset(HF_NAME, HF_CONFIG, split="test")
    seen = set()
    articles = []
    n_capped_prefix = 0
    for idx, row in enumerate(ds):
        if row["repo"] in seen:
            continue
        context = build_context(row["repo_snapshot"])
        cfile = row["completion_file"]
        gem_tokens = ng(context) + ng(cfile["content"])
        if gem_tokens > args.max_gemma_tokens:
            continue
        seen.add(row["repo"])

        lines = cfile["content"].split("\n")
        questions = []
        for cls in CLASSES:
            for line_no in sorted(row["completion_lines"].get(cls, [])):
                target = lines[line_no]
                if not target.strip():
                    continue  # blank target lines are unscorable
                prefix = "\n".join(lines[:line_no])
                if len(prefix) > args.max_prefix_chars:
                    prefix = prefix[-args.max_prefix_chars:]
                    prefix = prefix[prefix.find("\n") + 1:]  # start at a line boundary
                    n_capped_prefix += 1
                questions.append({
                    "question": INSTRUCTION.format(path=cfile["filename"], prefix=prefix),
                    "ruler_outputs": [target],
                    "answer_prefix": "```python\n",
                    "max_new_tokens": 64,
                    "task": f"lca_{cls}",
                    "line_no": line_no,
                    "question_unique_id": f"lca_{len(articles)}_{cls}_{line_no}",
                })
        articles.append({
            "article_id": f"lca_{len(articles)}_{row['repo']}",
            "title": f"{row['repo']} @ {row['commit_hash'][:10]} / {cfile['filename']}",
            "article": context,
            "questions": questions,
            "source": {
                "hf_dataset": HF_NAME, "hf_config": HF_CONFIG, "hf_index": idx,
                "repo": row["repo"], "commit_hash": row["commit_hash"],
                "completion_file": cfile["filename"],
                "context_tokens_qwen": nq(context), "context_tokens_gemma": ng(context),
                "completion_file_tokens_gemma": ng(cfile["content"]),
                "n_py_files": sum(1 for f in row["repo_snapshot"]["filename"] if f.endswith(".py")),
            },
        })
        print(f"[{len(articles):2d}] idx={idx:3d} {row['repo']:45} qwen={nq(context):7d} "
              f"gemma={gem_tokens:7d} questions={len(questions)}")
        if len(articles) == args.n_repos:
            break

    if len(articles) < args.n_repos:
        raise SystemExit(f"only {len(articles)} repos satisfy the cap; lower --n-repos or raise the cap")

    total_q = sum(len(a["questions"]) for a in articles)
    per_class = {c: sum(1 for a in articles for q in a["questions"] if q["task"] == f"lca_{c}") for c in CLASSES}
    meta = {
        "n_repos": len(articles), "total_questions": total_q, "questions_per_class": per_class,
        "max_gemma_tokens": args.max_gemma_tokens, "max_prefix_chars": args.max_prefix_chars,
        "n_prefixes_capped": n_capped_prefix,
        "qwen_tokens_median": sorted(a["source"]["context_tokens_qwen"] for a in articles)[len(articles) // 2],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "articles": articles}, f)
    print(json.dumps(meta, indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
