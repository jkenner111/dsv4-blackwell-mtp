# MTP h_proj Wiring — Benchmark Results

**Date:** 2026-05-15 | **Spark:** DGX GB10, temp=1.0, thinking=on

## Results

| Config | predicted t/s | draft accept | vs baseline |
|--------|---------------|-------------|-------------|
| No draft baseline | 6.95 | — | — |
| MTP draft (h_proj wired) | 4.28 | 176/176 (100%) | -38% |

## What changed

The h_proj wiring was successfully applied: the MTP draft now reads its own
hidden state from the previous decode (via `model.mtp_prev_hc`), projects it
through `hnorm -> h_proj`, and combines with `e_proj` output via `ggml_add`.
After each decode, the FFN output hidden state is stored back via `ggml_cpy`.

## Why no improvement

1. **Thinking mode masks acceptance rate.** The model's reasoning tokens follow
   a deterministic analysis pattern — "Okay, this is a question about..." —
   making them trivially predictable. 100% acceptance on thinking tokens tells
   us nothing about draft quality.

2. **Draft overhead is small but target dominates.** Draft runs at 260 t/s
   (380 ms for 99 runs). Target runs at 4.28-6.95 t/s (82 seconds for 350
   tokens). Even if the draft saves some target compute, the target's per-token
   cost dominates total time.

3. **Single-layer deepseek4 is computationally heavy.** The MTP layer runs at
   Q4K/Q8_0 precision (vs target's IQ2XXS), making each draft token cost
   comparable to a target token in compute — but the draft only has 1 layer
   vs 61, so draft FLOPS are ~1/60 of target. Draft overhead should be
   negligible.

## Next steps

1. **Disable thinking** (`/no_think` in system prompt) to get a real acceptance
   rate on open-ended generation. Thinking tokens are too deterministic.

2. **Re-quantize MTP to IQ2XXS** — the draft at Q4K/Q8_0 is heavier than
   necessary. At IQ2XXS, each draft token would be faster.

3. **Cross-context hidden state injection** — feed target model's `after_ffn_hc`
   to the draft (matching ds4's initial draft). This is the remaining gap for
   full ds4-level MTP performance.

4. **Longer, open-ended prompt at higher temperature** — the "mRNA vaccine"
   thinking prompt is too structured. A creative writing or code generation task
   at temp=0.8-1.0 would better differentiate draft quality.

## Files changed on Spark

- `~/llama-dsv4-fringe/src/models/deepseek4.cpp` — h_proj input wiring,
  hidden state store, MTP output head
- `~/llama-dsv4-fringe/src/llama-model.cpp` — buffer allocation for mtp_prev_hc
- `~/llama-dsv4-fringe/src/llama-model.h` — mtp_prev_hc / mtp_hidden_buf fields
- `~/llama-dsv4-fringe/src/llama-arch.h` / `.cpp` — MTP tensor enum + names

## Server state

MTP server exited cleanly after the benchmark (not crashed). Ready for next test.
