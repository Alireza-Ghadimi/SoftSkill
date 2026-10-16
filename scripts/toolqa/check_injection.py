"""Offline check of the softskill_init injection (no ReAct rollout).

For one question per gold skill, build SKIM's step-1 ReAct prompt and compare the
inputs_embeds generate_with_skills produces for full_text vs softskill_init=exact
(must be identical), then report vector counts/norms for the other inits.
Run from third_party/SKIM with scripts/toolqa on PYTHONPATH (see run_toolqa.sh).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "code"))
from skim_inference import generate_with_skills  # noqa: E402
from models.skim_model import SKIMModel  # noqa: E402
from skillrag_vendor.toolqa.react import ReActAgent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sra_root", required=True)
ap.add_argument("--llm", default=os.environ.get("QWEN3_8B_PATH"), required="QWEN3_8B_PATH" not in os.environ)
args = ap.parse_args()

sys.path.insert(0, str(Path(args.sra_root) / "src"))
from sragents.toolqa.fewshots import TOOLQA_EXAMPLES  # noqa: E402

corpus = {s["skill_id"]: s for s in json.load(open(Path(args.sra_root) / "data/bench/corpus/corpus.json"))
          if s["skill_id"].startswith("toolqa_")}
instances = json.load(open(Path(args.sra_root) / "data/bench/instances/toolqa.json"))
first = {}
for r in instances:
    first.setdefault(r["skill_annotations"][0], r)

model = SKIMModel(compressor_model="none", llm_model=args.llm, max_q=256, projector_layers=3, projector_hidden=4096,
                  torch_dtype=torch.bfloat16, skip_compressor=True)  # runner fallback defaults; projector unused
model.llm.to("cuda").eval()


def prompt(inst, mode):
    sid = inst["skill_annotations"][0]
    agent = ReActAgent(question=inst["question"], tools=None, model_generate=None, examples=TOOLQA_EXAMPLES,
                       method="naive" if mode == "naive" else "golden_skill", skills=[corpus[sid]["content"]], skill_ids=[sid], skill_mode=mode)
    return agent._build_prompt()


def embeds(inst, mode, init="", k=32):
    system, user = prompt(inst, mode)
    _, emb = generate_with_skills(model=model, corpus=corpus, system_text=system, user_text=user, skill_mode=mode,
                                  k=k, max_length=4096, max_new_tokens=1, do_sample=False, temperature=0.7,
                                  top_p=0.95, softskill_init=init)
    return emb[0].float()


all_equal = True
for sid, inst in sorted(first.items()):
    ft = embeds(inst, "full_text")
    ex = embeds(inst, "softskill_init", "exact")
    nv = embeds(inst, "naive")
    same = ft.shape == ex.shape and torch.equal(ft, ex)
    all_equal &= same
    row = [f"{sid}: full_text={ft.shape[0]} exact={ex.shape[0]} identical={same} naive={nv.shape[0]}"]
    tok_norm = model.llm.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
    for init, k in [("trunc", 32), ("chunkmean", 32), ("winpool", 0), ("random", 32), ("chunkmean", 256), ("chunkmean", 512)]:
        e = embeds(inst, "softskill_init", init, k)
        row.append(f"{init}{k or ''}:len={e.shape[0]}")
    print(" | ".join(row))
print(f"mean token-embedding norm {tok_norm:.3f}")
for init, k in [("trunc", 32), ("chunkmean", 32), ("winpool", 0), ("random", 32), ("chunkmean", 256), ("chunkmean", 512)]:
    from softskill_vectors import build_skill_vectors
    v = build_skill_vectors(model, "toolqa_009", corpus["toolqa_009"]["content"], init, k).float()
    print(f"toolqa_009 {init}{k or ''}: n={v.shape[0]} mean_norm={v.norm(dim=-1).mean():.3f}")
print("ALL EXACT == FULL_TEXT:", all_equal)
