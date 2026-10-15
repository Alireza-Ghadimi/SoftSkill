"""Score SKIM-runner ToolQA outputs with the official SRA-Bench ToolQA scorer.

Usage:
  score.py --sra_root <SR-Agents> --results a.jsonl [b.jsonl ...] [--out summary.json]

Each results file is one condition. Accuracy uses
sragents.evaluate.datasets.toolqa.evaluate (normalized exact match + numeric
and yes/no fallbacks; no LLM judge). Questions missing from a results file are
reported, not counted as wrong, so pilot and partial runs are flagged.
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# Easy/hard is not labelled in SRA-Bench. Each ToolQA domain has a lookup skill
# and an aggregation skill (matching ToolQA's easy/hard question sets), so the
# split is derived from the gold skill id.
EASY = {"toolqa_000", "toolqa_002", "toolqa_004", "toolqa_006", "toolqa_008", "toolqa_009", "toolqa_012"}
DOMAIN = {
    "toolqa_000": "scirex", "toolqa_001": "scirex", "toolqa_002": "flights", "toolqa_003": "flights",
    "toolqa_004": "airbnb", "toolqa_005": "airbnb", "toolqa_006": "yelp", "toolqa_007": "yelp",
    "toolqa_008": "coffee", "toolqa_010": "coffee", "toolqa_009": "agenda", "toolqa_011": "agenda",
    "toolqa_012": "dblp", "toolqa_013": "dblp",
}
ACTION_RE = re.compile(r"^Action (\d+): (.*)$", re.MULTILINE)
VALID_ACTION_RE = re.compile(r"^(\w+)\[(.+)\]$", re.DOTALL)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def unparseable_steps(scratchpad: str) -> tuple[int, int]:
    """(steps whose action is empty or not ActionType[arg], total steps)."""
    actions = ACTION_RE.findall(scratchpad)
    bad = 0
    for _, act in actions:
        act = act.strip()
        if not act:
            bad += 1
        elif not (act.startswith("PythonInterpreter[") and act.endswith("]")) and not VALID_ACTION_RE.match(act):
            bad += 1
    return bad, len(actions)


def score_file(path: Path, instances: dict, evaluate, skill_len: dict) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    seen = {}
    for r in rows:
        seen[r["instance_id"]] = r  # last write wins (resume duplicates)
    per = []
    for iid, r in seen.items():
        inst = instances[iid]
        res = evaluate(r["raw_output"], inst)
        bad, n_act = unparseable_steps(r["raw_output"])
        sid = inst["skill_annotations"][0]
        per.append({
            "instance_id": iid, "skill": sid, "correct": bool(res["correct"]),
            "extracted": res["extracted_answer"], "gt": str(inst["eval_data"]["answer"]),
            "n_steps": r.get("n_steps", n_act), "finished": bool(r.get("finished")),
            "halted": bool(r.get("halted")), "bad_steps": bad, "n_actions": n_act,
            "skill_len": skill_len.get(sid, 0),
        })
    n = len(per)
    k = sum(p["correct"] for p in per)
    lo, hi = wilson(k, n)

    def acc(sub):
        return round(100 * sum(p["correct"] for p in sub) / len(sub), 2) if sub else None

    by_skill = defaultdict(list)
    for p in per:
        by_skill[p["skill"]].append(p)
    return {
        "file": str(path), "n": n, "correct": k,
        "accuracy": round(100 * k / n, 2) if n else None,
        "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "avg_skill_len": round(sum(p["skill_len"] for p in per) / n, 1) if n else None,
        "avg_steps": round(sum(p["n_steps"] for p in per) / n, 2) if n else None,
        "finished_pct": round(100 * sum(p["finished"] for p in per) / n, 2) if n else None,
        "halted_pct": round(100 * sum(p["halted"] for p in per) / n, 2) if n else None,
        "unparseable_step_pct": round(100 * sum(p["bad_steps"] for p in per) / max(1, sum(p["n_actions"] for p in per)), 2),
        "easy_acc": acc([p for p in per if p["skill"] in EASY]),
        "hard_acc": acc([p for p in per if p["skill"] not in EASY]),
        "per_skill": {s: {"n": len(v), "acc": acc(v), "domain": DOMAIN.get(s), "easy": s in EASY}
                      for s, v in sorted(by_skill.items())},
        "details": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sra_root", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--skill_len", nargs="*", default=[],
                    help="Per file: 'none', 'text' (Qwen3 tokens of the gold skill) or an integer; default none")
    ap.add_argument("--tokenizer", default="/data/models/huggingface/qwen3-8b")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.sra_root) / "src"))
    # sragents.evaluate/__init__ imports every dataset scorer (BigCodeBench needs
    # tree_sitter); register bare packages so only the ToolQA scorer is loaded.
    import types
    for name in ("sragents.evaluate", "sragents.evaluate.datasets"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(Path(args.sra_root) / "src" / Path(*name.split(".")))]
        sys.modules[name] = pkg
    from sragents.evaluate.datasets.toolqa import evaluate

    instances = {r["instance_id"]: r for r in json.load(open(Path(args.sra_root) / "data/bench/instances/toolqa.json"))}
    corpus = None
    tok = None
    out = []
    for i, f in enumerate(args.results):
        spec = args.skill_len[i] if i < len(args.skill_len) else "none"
        skill_ids = sorted({r["skill_annotations"][0] for r in instances.values()})
        if spec == "none":
            lens = {}
        elif spec == "text":
            if corpus is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(args.tokenizer)
                corpus = {s["skill_id"]: s for s in json.load(open(Path(args.sra_root) / "data/bench/corpus/corpus.json"))
                          if s["skill_id"].startswith("toolqa_")}
            lens = {s: len(tok(corpus[s]["content"])["input_ids"]) for s in skill_ids}
        else:
            lens = {s: int(spec) for s in skill_ids}
        res = score_file(Path(f), instances, evaluate, lens)
        out.append(res)
        print(f"{Path(f).name}: n={res['n']} acc={res['accuracy']} ci95={res['ci95']} steps={res['avg_steps']} "
              f"fin={res['finished_pct']} halt={res['halted_pct']} bad={res['unparseable_step_pct']} "
              f"easy={res['easy_acc']} hard={res['hard_acc']} skill_len={res['avg_skill_len']}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
