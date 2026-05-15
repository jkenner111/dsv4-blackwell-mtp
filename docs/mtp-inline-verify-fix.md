# MoE-Aware Tiny-Batch Verification — Fix Plan

## Root cause confirmed

The speculative loop in llama.cpp does this for verification:
1. Collects draft tokens into a batch (e.g. [token_0, token_1, token_2, ...])
2. Calls `llama_decode(ctx_tgt, batch)` to verify all at once
3. For a single-token batch: standard decode, 127 ms/token
4. For a 2+ token batch: runs the FULL prefill path across all 61 layers

The MoE architecture makes small batches worse than sequential:
- Each token routes to different top-8 experts
- Expert matmuls run sequentially (no weight sharing across tokens)
- Fused DSv4 kernels materialize batch intermediates
- Result: 340 ms for a 2-token batch vs 254 ms for 2 sequential tokens

## Fix: Inline verification with single-token decode

Instead of batching N draft tokens and verifying all at once, verify
one draft token at a time using the standard decode path. Match against
the draft token, accept/reject, continue.

### File: tools/server/server-context.cpp (~line 395)

```cpp
    // REPLACE the batch-verify path:
    //
    //   common_batch_add(batch, sampled, pos0++, { this->id }, true);
    //   for (auto token : spec_draft) {
    //       common_batch_add(batch, token, pos0++, { this->id }, true);
    //   }
    //
    // WITH inline single-token verify:

    if (spec_draft.empty()) {
        common_batch_add(batch, sampled, pos0++, { this->id }, true);
    } else {
        // Inline verification: verify one draft token at a time
        // using single-token decode (faster than batched on MoE)
        int n_accepted = 0;

        // First, decode the sampled baseline token
        common_batch_add(batch, sampled, pos0++, { this->id }, true);

        int ret = llama_decode(ctx, batch);
        if (ret != 0 && ret != 1) break;

        // Verify each draft token one at a time
        for (auto draft_token : spec_draft) {
            // Get the target's predicted token
            llama_token target_token = common_sampler_sample(
                smpl.get(), ctx, -1, true);

            if (target_token == draft_token) {
                n_accepted++;
                // Commit: add to batch and decode
                batch.n_tokens = 0;
                common_batch_add(batch, draft_token, pos0++,
                    { this->id }, true);
                ret = llama_decode(ctx, batch);
                if (ret != 0 && ret != 1) break;
            } else {
                // Reject: target disagrees
                break;
            }
        }

        n_draft_total = spec_draft.size();
        n_draft_accepted = n_accepted;

        // Skip the already-accepted tokens in prompt state
        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(
            spec_draft.begin(),
            spec_draft.begin() + n_accepted);
    }
```

### Why this should help

| Operation | Before (batched) | After (inline) |
|-----------|-----------------|----------------|
| 1 token | 127 ms | 127 ms |
| 2 tokens | 340 ms (batch) | 254 ms (2×127) |
| 10 tokens | 1700 ms (batch) | 1270 ms (10×127) |

The inline path uses standard single-token decode for each verification,
avoiding the batched prefill path entirely. This matches ds4's approach:
verify one token at a time, but use the target's standard decode.

### Caveat

This loses the "verify all drafts and skip N accepted" optimization.
Instead of verifying N tokens in one batch pass and committing N,
we verify one at a time. For models where batched verification IS
faster (dense models, non-MoE), this would be a regression.

For DeepSeek V4 MoE specifically, batched verification is 2.5x slower
per token, so inline verification should be a net win.

### Alternative: MoE-aware batch verification

A more sophisticated fix would optimize the batched decode path for
MoE architectures: fuse expert dispatch across batch tokens, share
router computations, and use layer-major ordering (like ds4's
`verify_decode2_exact`) to avoid materializing full batch intermediates.
That's a ~200 line change across the CUDA kernels and graph builder.
