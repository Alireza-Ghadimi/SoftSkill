"""Fixed 200-question ToolQA pilot, stratified by gold skill.

Allocation is proportional to the full set (14 per skill, 18 for toolqa_010,
which has 130 questions), so pilot accuracy weights skills like the full 1,430.
Writes the id list into the repo and an instances dir the SKIM runner can read.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--instances", required=True, help="SRA-Bench data/bench/instances/toolqa.json")
ap.add_argument("--ids_out", required=True)
ap.add_argument("--instances_dir_out", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = json.load(open(args.instances))
by_skill = defaultdict(list)
for r in rows:
    by_skill[r["skill_annotations"][0]].append(r)

# Largest-remainder proportional allocation.
quota = {s: args.n * len(v) / len(rows) for s, v in by_skill.items()}
alloc = {s: int(q) for s, q in quota.items()}
for s in sorted(quota, key=lambda s: (-(quota[s] - alloc[s]), s))[: args.n - sum(alloc.values())]:
    alloc[s] += 1

rng = random.Random(args.seed)
picked = []
for s in sorted(by_skill):
    picked += rng.sample(sorted(by_skill[s], key=lambda r: r["instance_id"]), alloc[s])
picked.sort(key=lambda r: r["instance_id"])

json.dump({"seed": args.seed, "allocation": dict(sorted(alloc.items())),
           "instance_ids": [r["instance_id"] for r in picked]}, open(args.ids_out, "w"), indent=1)
out = Path(args.instances_dir_out)
out.mkdir(parents=True, exist_ok=True)
json.dump(picked, open(out / "toolqa.json", "w"), ensure_ascii=False, indent=1)
print(len(picked), dict(sorted(alloc.items())))
