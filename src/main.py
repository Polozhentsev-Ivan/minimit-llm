import argparse
import json
import os
from typing import Iterator, Dict, Any

from .llms import OpenAIModel, LlamaHFModel, BaseModel


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterator[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(diff: str) -> str:
    return (
        "### Task\n"
        "Write **one** concise, imperative Git commit message (≤ 12 words).\n"
        "Output **only** that single line.\n"
        "### Diff\n```diff\n"
        f"{diff}\n```"
    )
    

def truncate_lines(text: str, n: int) -> str:
    if n <= 0:
        return text
    return "\n".join(text.splitlines()[:n])


def get_model(kind: str, name: str) -> BaseModel:
    if kind == "openai":
        return OpenAIModel(model_name=name)
    if kind == "llama":
        return LlamaHFModel(repo_id=name)
    raise ValueError(kind)


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=["openai", "llama"], required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_lines", type=int, default=400)
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()

    model = get_model(args.model, args.model_name)
    dst_path = os.path.join(
        args.out_dir,
        f"{args.model_name.replace('/', '_')}-mcmd.jsonl",
    )

    rows_out = []
    for idx, row in enumerate(read_jsonl(args.dataset)):
        if args.limit and idx >= args.limit:
            break

        diff_raw = row["diff"].replace("<nl>", "\n")
        diff_use = truncate_lines(diff_raw, args.max_lines)
        prompt = build_prompt(diff_use)

        try:
            gen = model.generate(prompt)
        except Exception:
            diff_use = truncate_lines(diff_raw, args.max_lines // 2)
            gen = model.generate(build_prompt(diff_use))

        rows_out.append({
            "diff": row["diff"],
            "gold": row.get("commit_msg") or row.get("gold"),
            "generated": gen.strip(),
        })

    write_jsonl(dst_path, rows_out)


if __name__ == "__main__":
    run()
