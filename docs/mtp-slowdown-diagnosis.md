# MTP Slowdown Diagnosis — 4.5 t/s with MTP vs 9 t/s without

Root cause: llama.cpp's --model-draft runs the draft model as a black-box
independent decoder. ds4's MTP draft REQUIRES the target model's hidden state.
Without it, the draft predicts blind and each draft token costs MORE compute
than a main model token.

## What ds4 MTP actually does (from ds4.c:12721-12800)

The MTP forward pass takes THREE inputs, not one:

1. token       — the current draft token (from embed -> e_proj pathway)
2. prev_hc     — the TARGET model's after_ffn_hc (from hnorm -> h_proj pathway)
3. pos         — KV cache position

These two pathways are ADDED together before the decode layer:
  e_hc (from token embedding) + h_hc (from target hidden state) -> input_hc

Then: input_hc -> standard decode layer -> MTP output head -> logits

The h_proj pathway is NOT optional — it provides the context from the target
model. Without it, the MTP draft predicts "what token might plausibly come next"
rather than "what token the target model is actually driving toward."

## Why it's SLOWER (4.5 vs 9 t/s)

1. **BLIND DRAFT**: Without target hidden state (h_proj pathway), the MTP
   draft produces poor predictions. The acceptance rate on deterministic
   prompts is 100% (trivial), but on open-ended prompts the draft will diverge
   quickly from the target, wasting verification compute.

2. **DRAFT IS HEAVIER THAN TARGET**: The MTP GGUF uses Q4K for experts and
   Q8_0 for projections. The MAIN model uses IQ2XXS (2-bit) for experts.
   Each draft token runs matmul at 4-8 bits while the target runs at 2 bits.
   The draft layer does MORE compute per token than the main model.

   | Component | Main model quant | MTP draft quant | Relative cost |
   |-----------|-----------------|-----------------|---------------|
   | Expert up/gate | IQ2_XXS (2.06 bit) | Q4_K (4 bit) | ~4x |
   | Expert down | Q2_K (2 bit) | Q4_K (4 bit) | ~4x |
   | Attention projections | Q8_0 (8 bit) | Q8_0 (8 bit) | same |
   | HC/compressor/indexer | F16 | F32 | ~2x |

3. **OVERHEAD PER DRAFT TOKEN**: llama.cpp's speculative decoding runs
   draft_decode, then target_verify for each batch. The target verify is a
   batched prefill (cheap per token), but the draft decode is sequential
   (one forward pass per draft token). With the draft being computationally
   heavier per pass, the total time per generated token INCREASES unless
   acceptance rate is very high.

## What needs to change

### Short term (no C++ API changes):
- **Re-quantize the MTP draft** to IQ2XXS or lower, matching or beating
  the main model's speed. A 1-layer draft at IQ2XXS would be ~400 MB and
  run ~4x faster per forward pass.
- **Reduce draft count** (--draft-max 2 instead of 19). The overhead is
  linear in draft tokens, so fewer drafts = less waste.

### Medium term (C++ changes in Fringe210):
- **Wire the h_proj pathway** by passing the target model's after_ffn_hc
  into the draft model's context. Requires a new API to extract internal
  tensors from one llama_context and inject them into another.
- This matches ds4's actual architecture and should restore MTP acceptance
  rates to 70-85% on open-ended prompts.

### Long term:
- **Integrated MTP loader** — load the MTP model as a special draft that
  runs inside the target model's context, sharing the KV cache and hidden
  state directly. Eliminates the cross-context transfer overhead entirely.

## Why the 100% acceptance on "say hi" is misleading

The prompt "say hi in 5 words" with temperature=0 is a deterministic greedy
path. ANY draft model — even a random 2-layer transformer — would hit 100%
acceptance because there's only one correct token at each position. The
metric is saturated. A real test needs:
- Longer prompt (100+ tokens of context)
- Higher temperature (>= 0.7)
- Open-ended generation (not a constrained word-count task)

## Immediate action for the Spark

1. Try --draft-max 2 (in the server launch command) to reduce overhead
2. Test with a real prompt: "Write a 200-word explanation of mRNA vaccines"
   at temperature 0.7
3. Compare draft_n_accepted/draft_n ratio between --draft-max 2 and
   --draft-max 19
4. Post the results — we need the actual acceptance rate on a real prompt
   to know whether the draft is useful or just wasting compute
