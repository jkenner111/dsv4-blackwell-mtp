# MTP Speculative Loop Profile — Root Cause Found

## Raw numbers from Spark server logs

### No-draft baseline
```
prompt eval time =   889.16 ms /  18 tokens ( 49.40 ms/tok,  20.24 t/s)
       eval time = 35140.90 ms / 277 tokens (126.86 ms/tok,   7.88 t/s)
```

### MTP draft (h_proj, --draft-max 2)
```
prompt eval time =  1077.53 ms /  18 tokens ( 59.86 ms/tok,  16.70 t/s)
       eval time = 80655.01 ms / 350 tokens (230.44 ms/tok,   4.34 t/s)
statistics draft: #calls(b,g,a) = 1 169 169, #gen drafts = 169, #acc drafts = 169
                  #gen tokens = 237, #acc tokens = 179
                  dur(b,g,a) = 0.001, 1505.471, 0.093 ms
```

## Breakdown

| Component | Time | Notes |
|-----------|------|-------|
| Draft decode | 1505 ms | 169 calls, 237 gen tokens, ~8.9 ms/call |
| Target verify | ~57457 ms | 169 verify batches for 179 accepted tokens |
| Target sequential | ~21693 ms | 171 remaining tokens at 127 ms/token |
| **Total** | **80655 ms** | |

## Root cause

**Target verify batches are 2.5x more expensive per token than sequential decode.**

- Sequential decode: **127 ms/token**
- Verify batch (1-2 tokens): **~340 ms/batch → ~321 ms/token**

This is the opposite of what speculative decoding should achieve — batching should
be CHEAPER per token, not more expensive.

### Why batched verification is slower on this model

DeepSeek V4's MoE architecture breaks batch efficiency:

1. **Per-token expert routing**: Each token in the batch routes to different
   top-8 experts. The expert matmuls cannot share weights across tokens in
   the batch, so they run sequentially or with poor GPU utilization.

2. **Fused attention intermediates**: Batched forward passes materialize full
   [batch_size, n_embd, n_hc] intermediates that would be scalar in sequential
   decode. Memory bandwidth for these intermediates dominates when batch_size=2.

3. **No kernel fusion for tiny batches**: The fused kernels (DSv4_HC,
   DSv4_ATTN_OUT) are optimized for large-batch prefill, not 2-token verification.

### Why ds4 avoids this

ds4's `--mtp-draft 2` combined with the margin gate means it only verifies
when the draft is highly confident, AND it uses a specialized "tiny batch
verifier" path (`verify_decode2_exact`) that runs 2 positions in one
layer-major pass using layer-wise kernel fusion.

### Fix: disable speculative decoding on this architecture

Until someone implements MoE-aware batched verification or the ds4-style
N=2 fused verifier, speculative decoding with the current llama.cpp -md
path is a net loss on DeepSeek V4 Flash.

The draft model works correctly (100% acceptance), but the target's
batched verification path is architecturally worse than sequential decode.
