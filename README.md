# SoftSkill on Qwen3-8B

Reproduction and extension of **SoftSkill: Behavioral Compression for Contextual Adaptation**
(Tao et al., 2026, [arXiv 2606.20333](https://arxiv.org/abs/2606.20333)) with the authors' code,
changing the target model from Qwen3.5-4B to **Qwen3-8B**.

A *soft skill* is a short sequence of learned input embeddings (32 vectors here), inserted into the
prompt of a frozen LLM in place of a long natural-language skill document. This repo asks three questions:

1. Does the paper's Table 1 hold on Qwen3-8B? (SearchQA, LiveMath)
2. How much of the gain comes from training, and how much from the initial vectors (p0)?
3. Do **untrained** SoftSkill vectors help an agent on ToolQA, in the setting of
   [SKIM](https://github.com/bebr2/SKIM)'s Table 2?

The full write-up, with every number, caveat and decision, is in
**[`softskill_qwen3_report.md`](softskill_qwen3_report.md)**.

## Headline results

**SearchQA (Table 1, test-hard accuracy %, 3 seeds): reproduces.**

| Method | Skill tokens | Qwen3-8B (this repo) | Qwen3.5-4B (paper) |
|---|---|---|---|
| No skill | 0 | 67.9 | 68.1 |
| Hard skill (SkillOpt text) | 1,995 | 71.7 | 71.2 |
| **SoftSkill** (prompt_start) | 32 | **74.2 ± 0.3** | 76.4 |

32 trained vectors beat a ~2k-token text skill on a second model. Untrained p0 alone scores
66.9–67.7, no better than no skill, so the whole gain comes from training.

**LiveMath: upstream setting is not a valid comparison on Qwen3-8B.** With the upstream 16-token
decoding cap, every baseline is cut off mid-reasoning and scores 0. With `max_new_tokens=2048`
(3 seeds): hard skill 26.1 > untrained windowed-pool p0 23.7 > no skill 19.9 (report §6, §8.4).

**ToolQA pilot (200 of 1,430 questions, frozen Qwen3-8B, SKIM ReAct runner; report §9).**

| Condition | Accuracy % (95% CI) | Easy | Hard |
|---|---|---|---|
| No skill (naive) | 30.5 [24.5, 37.2] | 46.9 | 14.7 |
| Gold skill as full text | **50.5** [43.6, 57.4] | 61.2 | 40.2 |
| Untrained chunk-mean, k = 16 | 28.5 [22.7, 35.1] | 42.9 | 14.7 |
| Untrained chunk-mean, k = 32 | 30.5 [24.5, 37.2] | 42.9 | 18.6 |
| Untrained chunk-mean, k = 64 | in progress | | |

Across all three studies, untrained pooled skill vectors match no skill; only trained vectors or
the full text skill help.

## Repository layout

```
softskill_qwen3_report.md   full report (§1–7 Table 1, §8 follow-ups, §9 ToolQA)
memory.md                   short running status log (what is in flight, next steps)
configs/README.md           effective hyperparameters (upstream YAMLs are used unchanged)
patches/                    our changes to the upstream submodules, applied by the setup scripts
  qwen3.patch                 opt-in gradient checkpointing (needed to fit Qwen3-8B on 40 GB)
  pooled_init.patch           init strategies text_max_pool / text_mean_pool
  window_pooled_init.patch    init strategy text_mean_pool_window
  spreadsheet_docker_sandbox.patch  opt-in Docker sandbox for SpreadsheetBench code
  skim/toolqa.patch           SKIM ToolQA runner: base-model mode, untrained SoftSkill injection
scripts/
  setup_env.sh                create venvs, init submodules, apply patches
  prepare_data.sh             build SearchQA / LiveMath official splits
  prepare_livemath_pinned.py  LiveMath split pinned to the paper's manifest
  run_one.sh                  one SoftSkill run (task, placement, seed, mode, model, GPU)
  run_with_hard_skill.py      hard-skill (text) baseline
  launch_all.sh               queue all Table 1 runs over a set of GPUs
  launch_livemath_*.sh        LiveMath max_new_tokens=2048 queues (vLLM prompt-embeds server)
  mirror_logs.sh, summarize.py  copy logs into logs/ and build logs/summary.{json,md}
  toolqa/                     ToolQA experiment (see below)
docker/spreadsheet_sandbox/ sandbox image for model-generated SpreadsheetBench code
data/toolqa/pilot_ids.json  fixed 200-question ToolQA pilot, stratified by gold skill
logs/                       run logs, summary.{json,md}, agentic trajectories, toolqa/pilot_summary.json
third_party/
  SoftSkill                   submodule: github.com/xijia-tao/SoftSkill (pinned 4fc5300)
  SKIM                        submodule: github.com/bebr2/SKIM (pinned 902de6d)
```

Large files (datasets, checkpoints, raw results, virtualenvs) live outside git under
`/data/a84460786/softskill_qwen3/`. Set `SOFTSKILL_ROOT` to use another location.

## Setup

Requirements: Linux, NVIDIA GPUs with 40 GB or more (A100 40 GB used here), CUDA 13, and Qwen3-8B
weights (default path `/data/models/huggingface/qwen3-8b`).

```bash
git clone --recurse-submodules <this repo>
cd softskill-qwen3
scripts/setup_env.sh        # venv under $SOFTSKILL_ROOT, applies patches/*.patch to third_party/SoftSkill
scripts/prepare_data.sh     # SearchQA and LiveMath splits + skill artifacts
```

Main versions: torch 2.14 (cu130), transformers 5.17, vLLM 0.31.0 (separate venv, LiveMath
long-budget runs only).

## Running

**Table 1 (SearchQA, LiveMath).**

```bash
nohup scripts/launch_all.sh "5 6" > $SOFTSKILL_ROOT/results/logs/launch_all.out 2>&1 &   # one worker per GPU
scripts/run_one.sh searchqa prompt_start 1 train qwen3_8b 5                              # or a single run
scripts/mirror_logs.sh                                                                    # refresh logs/summary.*
```

`run_one.sh` modes: `train` (SoftSkill), `init` (untrained p0), `init_maxpool`, `init_meanpool`,
`init_winpool`. Runs whose `summary.json` already exists are skipped.

**ToolQA (untrained SoftSkill in SKIM's runner).** Needs a clone of
[SR-Agents](https://github.com/oneal2000/SR-Agents) (SRA-Bench data and scorer) and ToolQA's
`external_corpus` under `/data/a84460786/softskill_qwen3/toolqa/data/`.

```bash
scripts/toolqa/setup_skim.sh                                      # apply patches/skim/toolqa.patch
scripts/toolqa/run_toolqa.sh <gpu> pilot naive naive              # split: smoke | pilot | full
scripts/toolqa/run_toolqa.sh <gpu> pilot full_text full_text
scripts/toolqa/run_toolqa.sh <gpu> pilot softskill_init chunkmean_k32 --softskill_init chunkmean --k 32
scripts/toolqa/score.py --sra_root <SR-Agents> --results <results/pilot/*.jsonl> --out summary.json
```

Runs resume automatically (`--resume`, one result line per question). `--softskill_init` also accepts
`exact` (bit-identical to full text, used as a check), `trunc`, `winpool` and `random`.

## Known caveats

- LiveMath is small (18 validation, 124 test items), so expect about ±4 points of noise. The
  16-token upstream budget truncates Qwen3-8B's reasoning (report §6).
- DocVQA (vision) cannot run on text-only Qwen3-8B. The LoRA baseline was not run.
- ToolQA numbers are a 200-question pilot (±6.5 points). SKIM's ReAct parser keeps only the first
  `Action` line, as in SR-Agents.
- Agentic tasks (ALFWorld, SpreadsheetBench): only no-skill smoke tests so far (report §8.2).

## Credits

Upstream code: [xijia-tao/SoftSkill](https://github.com/xijia-tao/SoftSkill) and
[bebr2/SKIM](https://github.com/bebr2/SKIM). Benchmark data: SearchQA, LiveMathematicianBench,
[SR-Agents / SRA-Bench](https://github.com/oneal2000/SR-Agents) and ToolQA. All credit for the
methods goes to their authors. This repo only adds patches, launch scripts and the evaluation report.
