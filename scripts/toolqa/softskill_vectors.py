"""Untrained SoftSkill vectors (p0, no training) for SKIM's <skill> slot.

Imported by the patched SKIM runner (skill_mode=softskill_init). Every vector is
built from the frozen Qwen3-8B input embedding table, so nothing here is learned.

Inits (--softskill_init, with --k vectors where it applies):
  exact     every skill token, embedded in its full-text context (condition A)
  trunc     SoftSkill default `text` init: first k tokens, tiled if shorter (B)
  chunkmean mean over k contiguous, near-equal chunks of the skill tokens (C, E)
  winpool   SoftSkill text_mean_pool_window, window 8, stride 6 (D; k ignored)
  random    Gaussian directions rescaled to the per-vector norms of chunkmean (F)

`exact` is handled by the caller (see skim_inference.generate_with_skills),
because it needs the token span of the skill inside the full user string.
"""
from __future__ import annotations

import zlib

import torch

WINPOOL_WINDOW = 8
WINPOOL_STRIDE = 6


def text_window_pool_starts(n_tokens: int, window: int, stride: int) -> list[int]:
    """Copied from third_party/SoftSkill skillopt/softprefix/model.py (window_pooled_init.patch)."""
    if n_tokens <= window:
        return [0]
    return list(range(0, n_tokens - window + stride, stride))


def _chunk_mean(embeds: torch.Tensor, k: int) -> torch.Tensor:
    if embeds.shape[0] < k:
        raise ValueError(f"chunkmean needs >= k tokens (got {embeds.shape[0]} for k={k})")
    return torch.stack([chunk.mean(dim=0) for chunk in torch.tensor_split(embeds, k, dim=0)])


def build_skill_vectors(model, skill_id: str, skill_text: str, init: str, k: int) -> torch.Tensor:
    """Return [n_vectors, hidden] in the LLM embedding dtype/device."""
    tokenizer = model.llm_tokenizer
    emb_layer = model.llm.get_input_embeddings()
    ids = tokenizer.encode(skill_text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty skill text for {skill_id}")
    ids_t = torch.tensor(ids, device=emb_layer.weight.device, dtype=torch.long)
    with torch.no_grad():
        embeds = emb_layer(ids_t).float()
        if init == "trunc":
            # Same as SoftPrefixCausalLM.initialize_from_text: tile, then cut to k rows.
            repeats = (k + embeds.shape[0] - 1) // embeds.shape[0]
            vecs = embeds.repeat((repeats, 1))[:k]
        elif init == "chunkmean":
            vecs = _chunk_mean(embeds, k)
        elif init == "winpool":
            starts = text_window_pool_starts(embeds.shape[0], WINPOOL_WINDOW, WINPOOL_STRIDE)
            vecs = torch.stack([embeds[s : s + WINPOOL_WINDOW].mean(dim=0) for s in starts])
        elif init == "random":
            norms = _chunk_mean(embeds, k).norm(dim=-1, keepdim=True)
            gen = torch.Generator(device="cpu").manual_seed(zlib.crc32(skill_id.encode()))
            noise = torch.randn(k, embeds.shape[1], generator=gen).to(embeds.device)
            vecs = noise / noise.norm(dim=-1, keepdim=True) * norms
        else:
            raise ValueError(f"unknown softskill_init {init!r}")
    return vecs.to(emb_layer.weight.dtype)
