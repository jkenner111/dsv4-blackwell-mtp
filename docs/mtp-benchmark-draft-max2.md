# MTP Benchmark Results — Full Comparison

**Date:** 2026-05-15 | **Spark:** DGX GB10 | **Temp:** 1.0

## Results

| Run | pred t/s | draft accept | vs baseline |
|-----|----------|-------------|-------------|
| No-draft, /no_think | **7.88** | — | — |
| MTP h_proj, /no_think, draft-max 16 | 4.39 | 163/163 (100%) | −44% |
| MTP h_proj, /no_think, draft-max 2 | 4.39 | 179/179 (100%) | −44% |

`--draft-max 2` had no effect — llama.cpp generates the full speculative window regardless.

## Key insight from DwarfStar 4 documentation

The antirez ds4 engine documents MTP as:
- "provides at most a slight speedup, not a meaningful generation-speed win"
- Uses `--mtp-draft 2` (only 2 draft tokens per cycle)
- Applies a confidence gate (`--mtp-margin`, default 3.0): skips verification when draft logit margin is low
- Greedy decoding only (`sample_argmax`)

Our llama.cpp speculative decoding generates 163-179 draft tokens per cycle with no margin gate. Each draft token costs ~3.8ms (260 t/s draft throughput), but the target verification path costs ~200ms per batch, and with 100% acceptance, the overhead dominates.

## Path forward

1. **Profile speculative loop** — measure where the 44% overhead goes (draft decode vs target verify vs context switch)
2. **Margin gate** — implement logit-margin check on draft tokens to skip uncertain verifications
3. **ds4's N=2 batch verifier** — the ds4 code path that verifies 2 tokens in one layer-major pass is faster than sequential decode per token
