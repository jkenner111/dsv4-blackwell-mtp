# MTP Port Gap Analysis: ds4 (Metal) to Fringe210 (CUDA)

**Date:** 2026-05-15 | **Target:** NVIDIA GB10 (SM121, Blackwell)
**Source:** antirez/ds4 | **Target repo:** Fringe210/llama.cpp-deepseek-v4-flash-cuda

---

## What ds4 MTP does (Metal)

MTP is a separate single-layer GGUF model loaded alongside the main model via --mtp.
Target hidden state (cur_hc) feeds into the MTP draft model.

### MTP forward pass (9 steps)
1. Embed token (base_model.token_embd) -> RMSNorm (mtp.0.enorm) -> MatMul (mtp.0.e_proj)
2. Repeat to HC count
3. RMSNorm prev_hc (mtp.0.hnorm) -> MatMul (mtp.0.h_proj)
4. Add e_proj + h_proj -> input_hc
5-8. Standard decode layer (HC split->attn->combine; HC split->MoE->combine)
9. MTP output head (hc_head_fn->scale/base->HC weighted sum->norm->lm_head)

### Key design
- Separate KV cache (mtp_raw_cache)
- Hidden state injection: Target cur_hc feeds MTP draft (the only coupling)
- Recursive drafting with alternating hidden state buffers
- Target weight sharing: token_embd + lm_head from target model

### Custom ops: ALL FIVE DSv4 ops used. NO NEW OPS.

### Speculate->verify loop
1. Target decode -> logits
2. Free verify draft[0] against target argmax
3. Recursive draft for [2..draft_cap]
4. Verify suffix (batched prefill, accept consecutive matches)
5. Commit accepted, truncate MTP cache counter

---

## What Fringe210 does (CUDA)

Upstream llama.cpp speculative decoding (examples/speculative/speculative.cpp):
TWO independent llama_context objects, draft runs tree-based decode.
No hidden-state injection mechanism.

### CUDA kernels (ggml/src/ggml-cuda/dsv4.cu)
DSV4_HC_SPLIT_SINKHORN, DSV4_HC_EXPAND, DSV4_HC_WEIGHTED_SUM, DSV4_ROPE_TAIL, DSV4_FP8_KV_QUANTIZE -- all Complete.
Plus all standard ggml CUDA kernels.

---

## Kernel reuse: HIGH / MEDIUM / LOW

### HIGH (direct reuse)
All five DSv4 ops + MUL_MAT, RMS_NORM, FLASH_ATTN_EXT, SOFT_MAX, ADD, GET_ROWS, REPEAT

### MEDIUM
Tree-based speculative decoding in examples/speculative/ -- works with any draft model

### LOW
Direct hidden state injection -- no mechanism in llama.cpp

---

## New CUDA kernels: NONE

---

## Plumbing changes

1. Architecture entry (MODERATE): mtp.0.* vs blk.0.* tensor naming
2. MTP graph builder (MODERATE): embedding/hidden-state projection, MTP output head
3. Dual-model integration (HARD): target hidden state -> draft model
4. Separate KV cache (MODERATE): automatic if separate llama_context
5. Batched suffix verification (MODERATE): llama_batch API supports multi-token
6. Session management (MODERATE): hidden state alternation, cache rollback

---

## Architecture compatibility

MTP GGUF will NOT load via -md. Three reasons:
1. Tensor naming: mtp.0.* vs blk.0.*
2. No deepseek4-mtp architecture variant
3. No hidden-state injection mechanism in llama.cpp

---

## Recommended approach

| Step | Description | Est. | Complexity |
|------|-------------|------|------------|
| 1 | Architecture + tensor loading | 1-2 days | MODERATE |
| 2 | MTP graph builder | 2-3 days | MODERATE |
| 3 | Hidden state plumbing | 2-3 days | HARD |
| 4 | Speculative loop integration | 2-3 days | MODERATE |
| 5 | Performance tuning | 2-4 days | MODERATE |
| 6 | Testing | 2-3 days | MODERATE |
| **Total** | | **11-18 days** | |

Two-day quick path: Convert GGUF (rename tensors), use as independent draft model.
Two-week full port: Hidden-state injection for ds4-level MTP performance.

---

## Summary

Plumbing/integration challenge, not CUDA kernel development.
No new GPU kernels needed. Hard parts: hidden state injection, tensor name mapping, graph builder.
