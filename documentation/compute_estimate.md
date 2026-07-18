# Compute Budget: Geometry of Morality Pilot

## Baseline (Marks & Tegmark, 2023)
- Paper doesn't publish GPU-hours — App. H covers dataset construction, not compute setup.
- Activation extraction is **forward-pass only** (no training, no generation) — one pass per example, caching the residual stream at one token position.
- Dataset scale precedent: `cities` = 1,496 rows; largest dataset (`counterfact_true_false`) = 31,960 rows.
- Probing (LR/mass-mean/CCS) and causal interventions run on cached vectors — negligible added GPU cost.

## Storage

**Kept long-term (activations only, ~6,000-row pilot, all layers, fp32):**

| Model | Hidden dim | Layers | Storage |
|---|---|---|---|
| 7B | 4,096 | 32 | ~2.9 GB |
| 13B | 5,120 | 40 | ~4.7 GB |
| 70B | 8,192 | 80 | ~15.7 GB |

- fp16 halves these 

**Needed on the GPU instance during the run:**

| Component | Size |
|---|---|
| Model weights (bf16) — 7B / 13B / 70B | ~14 GB / ~26 GB / ~140 GB |
| Env/CUDA/dependencies | ~5–10 GB |
| Cached activations | ~3–16 GB (depends on model) |
| Datasets/code/misc | <1 GB |
| **Total** | **~25 GB (7B) / ~35–40 GB (13B) / ~155–165 GB (70B)** |

- Rent disk with headroom:
    - ~40 GB (7B) 
    - ~50–60 GB (13B)
    -  ~180 GB (70B)

## GPU recommendation per model (Vast.ai)

| Model | Weights (bf16) | Recommended GPU | Vast.ai rate |
|---|---|---|---|
| 7B | ~14 GB | 1× A100 40GB (PCIe) | ~$0.40/hr |
| 13B | ~26 GB | 1× A100 80GB (SXM4) | ~$0.67–1.10/hr |
| 70B | ~140 GB | 2× A100 80GB (SXM4, tensor-parallel) | ~$1.34–2.20/hr (combined) |

- 7B fits easily on the cheaper 40GB card 
- 13B needs 80GB for comfortable headroom (weights + activations + batch overhead).
- 70B's ~140 GB of weights doesn't fit on one GPU at any size

## Time & cost

| Model | GPU setup | Time (6,000 examples) | Est. cost per run (extraction only) |
|---|---|---|---|
| 7B | 1× A100 40GB | < 10 min | ~$0.05–0.10 |
| 13B | 1× A100 80GB | < 15 min | ~$0.15–0.25 |
| 70B | 2× A100 80GB | 15–45 min | ~$0.35–1.65 |



**How Vast.ai actually bills:** per-second, not per-hour — `cost = hourly_rate × (seconds_used / 3600)`. No minimum charge on on-demand instances. (Storage bills separately and continues while an instance exists, even if stopped — only stops when you delete the instance.)

**Cost per full pass, all three models, worst-case (upper-bound) rates:**

| Model | Rate (upper bound) | Time (setup + extraction) | Seconds | Calculation | Cost |
|---|---|---|---|---|---|
| 7B | $0.40/hr | 25 min | 1,500 | $0.40 × (1500/3600) | $0.167 |
| 13B | $1.10/hr | 30 min | 1,800 | $1.10 × (1800/3600) | $0.550 |
| 70B | $2.20/hr (2× A100) | 70 min | 4,200 | $2.20 × (4200/3600) | $2.567 |
| **Total per full pass (all 3 models)** | | | | | **$3.28** |

**Final plan: one active week (instances stopped-but-not-deleted), then delete with data saved**

Persistent storage bills continuously while an instance exists 

Formula: `storage_cost = disk_GB × $0.15/GB/month × (days_kept/30)`.

*Storage cost for a 7-day active window, all three instances kept alive:*

| Model | Disk | Daily cost | × 7 days |
|---|---|---|---|
| 7B | ~40 GB | $0.20/day | $1.40 |
| 13B | ~60 GB | $0.30/day | $2.10 |
| 70B | ~180 GB | $0.90/day | $6.30 |
| **Total storage (7 days)** | | | **$9.80** |

*Compute cost during that week (3 full passes — pilot, one revision, one causal-intervention pass):*

3 × $3.28/pass = **$9.84**

**Total: $9.80 (storage) + $9.84 (compute) = $19.64**

- 70B's disk is the single biggest cost driver even over just one week (~$6.30 of the $9.80) — if trimming margin, delete that one first.

## Decision: run all three sizes (7B, 13B, 70B)

- Mirrors the paper's own approach — they compare linear structure across 7B/13B/70B, since the key finding is that structure gets *more abstract with scale*. Running all three lets you see if the same scaling trend holds for morality.
- With the right GPU picked per model, the finalized plan — one active week with instances stopped-but-not-deleted, 3 full passes, then delete — comes to **$19.64** ($9.80 storage + $9.84 compute)
- **Still start with 13B first** as the quick sanity check (cheapest single-GPU setup, fastest to debug your pipeline on). Once that's working, run 7B and 70B using the same pipeline within the same active week.
- 70B needs 2 GPUs (tensor-parallel) and a much larger weight download (~140 GB) — expect it to be the slowest and most fiddly of the three, so leave more buffer time for it. Its disk is also the biggest storage cost driver during the active week.

## Summary

- **Models:** 7B, 13B, 70B — start with 13B, then run 7B and 70B
- **GPUs:** A100 40GB (7B), A100 80GB (13B), 2× A100 80GB (70B) — all via Vast.ai
- **Cost:** $19.64 total — $9.80 storage (7 active days, all 3 instances stopped-but-not-deleted) + $9.84 compute (3 full passes)
- **Timeline:** one active week, then delete instances (data/graphs saved elsewhere) — no further storage cost through project end (Aug 11)
- **Storage:** ~2.9–15.7 GB kept per model; up to ~180 GB disk needed during the 70B run