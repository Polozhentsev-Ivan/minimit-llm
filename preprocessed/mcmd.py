import os
import json
import argparse


def process(root_dir: str, output_path: str, encoding: str = "utf-8", limit: int | None = None, languages: list[str] | None = None) -> None:
    written = 0
    lang_set = set(languages) if languages else None
    with open(output_path, "w", encoding=encoding) as out_f:
        for language in sorted(os.listdir(root_dir)):
            if lang_set and language not in lang_set:
                continue
            lang_path = os.path.join(root_dir, language)
            if not os.path.isdir(lang_path):
                continue
            for strategy in sorted(os.listdir(lang_path)):
                strat_path = os.path.join(lang_path, strategy)
                if not os.path.isdir(strat_path):
                    continue
                for split in ("train", "valid", "test"):
                    base = os.path.join(strat_path, f"{split}")
                    diff_path = f"{base}.diff.txt"
                    if not os.path.exists(diff_path):
                        continue
                    msg_path = f"{base}.msg.txt"
                    repo_path = f"{base}.repo.txt"
                    sha_path = f"{base}.sha.txt"
                    time_path = f"{base}.time.txt"
                    with open(diff_path, encoding=encoding) as diff_f, open(msg_path, encoding=encoding) as msg_f, open(repo_path, encoding=encoding) as repo_f, open(sha_path, encoding=encoding) as sha_f, open(time_path, encoding=encoding) as time_f:
                        for diff, msg, repo, sha, time in zip(diff_f, msg_f, repo_f, sha_f, time_f):
                            obj = {
                                "diff": diff.rstrip("\n"),
                                "commit_msg": msg.rstrip("\n"),
                                "repo": repo.rstrip("\n"),
                                "sha": sha.rstrip("\n"),
                                "time": time.rstrip("\n"),
                                "language": language,
                                "split": split,
                                "strategy": strategy,
                            }
                            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            written += 1
                            if limit is not None and written >= limit:
                                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--languages", type=str)
    args = parser.parse_args()
    langs = [x.strip() for x in args.languages.split(",") if x.strip()] if args.languages else None
    process(args.input, args.output, limit=args.limit, languages=langs)


if __name__ == "__main__":
    main()
