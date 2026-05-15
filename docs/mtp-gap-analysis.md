# MTP Port Gap Analysis: ds4 (Metal) to Fringe210 (CUDA)

**Date:** 2026-05-15
**Target hardware:** NVIDIA GB10 (SM121, Blackwell)
**Source:** antirez/ds4 (Metal inference engine)
**Target repo:** Fringe210/llama.cpp-deepseek-v4-flash-cuda

---

## What ds4 MTP does (Metal)

### Architecture overview

The MTP component in ds4 is a **separate single-layer GGUF model** loaded alongside the main model via the `--mtp` flag.

TARGET MODEL (61 layers) passes hidden state (cur_hc) to MTP DRAFT MODEL (1 layer).
The draft model: embed(t0) -> e_proj + prev_hc -> h_proj -> input_hc -> standard decode layer -> MTP output head -> draft logits.

### MTP forward pass: 9 steps

1. Embed current draft token (base_model.token_embd)
2. RMSNorm embed (mtp.0.enorm)
3. MatMul: embed -> HC dim (mtp.0.e_proj)
4. Repeat to HC count
5. RMSNorm per-HC prev hidden state (mtp.0.hnorm)
6. MatMul: prev HC -> HC dim (mtp.0.h_proj)
7. Add: e_proj + h_proj -> input_hc
8. Standard decode layer (same as main model: HC split->attn->combine; HC split->MoE->combine)
9. MTP output head (hc_head_fn->scale/base->HC weighted sum->norm->lm_head)

### Key design decisions

1. Separate KV cache: MTP has its own raw SWA cache.
2. Hidden state injection: Target model's cur_hc feeds into the MTP draft.
3. Recursive drafting: Draft tokens feed back, alternating between mtp_state_hc and mtp_next_hc.
4. Target weight sharing: MTP output head uses target's token_embd and lm_head.

### Custom ops used by MTP

ALL FIVE DSv4 ops, exactly as main model: hc_split_sinkhorn, hc_expand, hc_weighted_sum, rope_tail, fp8_kv_quantize.
NO NEW CUSTOM OPS are introduced by MTP.

### Speculate -> verify -> accept/reject loop

1. TARGET DECODE -> target logits
2. FREE VERIFY draft[0]: argmax(target logits) == draft_token? No match -> return only first_token.
3. RECURSIVE DRAFT: for d in [2..draft_cap]: MTP draft(tokens[d-1], prev_hc) -> argmax -> tokens[d]
4. VERIFY SUFFIX (batched prefill): target model verify_suffix, accept consecutive matches, reject at first mismatch.
5. COMMIT: Keep accepted tokens, truncate MTP raw cache counter.

Verification modes: Tiny batch (N=2 default), Strict mode (sequential per token), Margin check.

---

## What Fringe210 speculative decoding does (CUDA)

Uses upstream llama.cpp standard speculative decoding from examples/speculative/speculative.cpp:
- TWO independent llama_context objects (ctx_tgt, ctx_dft)
- Each has its own KV cache, memory, graph
- Draft model runs tree-based speculative decoding via llama_decode()
- No DSv4-specific modifications to the speculative loop
- Vocabulary compatibility checked
- NO mechanism to inject target hidden state into draft model

### Existing CUDA kernels (all in ggml/src/ggml-cuda/dsv4.cu)

| GGML op | Status |
|---------|--------|
| GGML_OP_DSV4_HC_SPLIT_SINKHORN | Complete |
| GGML_OP_DSV4_HC_EXPAND | Complete |
| GGML_OP_DSV4_HC_WEIGHTED_SUM | Complete |
| GGML_OP_DSV4_ROPE_TAIL | Complete |
| GGML_OP_DSV4_FP8_KV_QUANTIZE | Complete |

Plus all standard ggml CUDA kernels (MUL_MAT, RMS_NORM, FLASH_ATTN_EXT, etc.)

---

## Existing CUDA kernels MTP can reuse

### HIGH (direct reuse)
DSV4_HC_SPLIT_SINKHORN, DSV4_HC_EXPAND, DSV4_HC_WEIGHTED_SUM, DSV4_ROPE_TAIL, DSV4_FP8_KV_QUANTIZE, MUL_MAT, RMS_NORM, FLASH_ATTN_EXT, SOFT_MAX, ADD, GET_ROWS, REPEAT

### MEDIUM
Tree-based speculative decoding in examples/speculative/ - works with any draft model, but ds4's hidden-state injection is fundamentally different.

### LOW
Direct hidden state injection - llama.cpp has no mechanism to pass hidden state between contexts.

---

## New CUDA kernels required

NONE. All ops already ported. Challenge is in architecture definition, tensor loading, and integration plumbing.

---

## Plumbing changes required

### 1. New model architecture entry (MODERATE)
MTP GGUF uses mtp.0.* tensor names; Fringe210 expects blk.0.*.
Options: A) New deepseek4-mtp architecture, B) Tensor name remapping, C) Converted GGUF.

### 2. MTP-specific model graph (MODERATE)
MTP graph differs: 1 layer, embedding projection step, hidden state projection step, MTP output head.

### 3. Dual-model integration (HARD)
Target hidden state must feed into draft model. No existing mechanism in llama.cpp.

### 4. Separate KV cache (MODERATE)
Automatic if loaded as separate llama_context.

### 5. Batched suffix verification (MODERATE)
llama_batch API already supports multi-token batches.

### 6. Session management for recursive drafting (MODERATE)
Hidden state alternation, cache counter rollback.

---

## Architecture compatibility

Will MTP GGUF load via -md? NO. Reasons:
1. Tensor naming mismatch (mtp.0.* vs blk.0.*)
2. No deepseek4-mtp architecture variant
3. No hidden-state injection mechanism

---

## Recommended approach

Step 1: Architecture and tensor loading (1-2 days)
Step 2: MTP model graph builder (2-3 days)
Step 3: Hidden state plumbing (2-3 days) - HARD
Step 4: Speculative loop integration (2-3 days)
Step 5: Performance tuning (2-4 days)
Step 6: Testing and validation (2-3 days)

Total: 11-18 days.

Two-day quick path: Convert GGUF (rename tensors), use as independent draft model.
Two-week full port: Includes hidden-state injection for ds4-level performance.

---

## Summary

Primarily a plumbing/integration challenge, not CUDA kernel development.
No new GPU kernels needed. Hard parts: hidden state injection, tensor name mapping, graph builder.
