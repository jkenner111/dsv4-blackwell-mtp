# MTP h_proj Self-Hidden-State Wiring

## Problem

llama.cpp's -md speculative decoding has no cross-context API. Can't pass
target hidden state to draft model. ds4's MTP uses target.after_ffn_hc as
prev_hc for the first draft token.

## Solution: Self-hidden-state approach

Store the draft model's own after_ffn_hc between decode calls and use it
as prev_hc. ds4 already does this for recursive drafting (tokens 2..N use
the draft's own hidden state). The first token gets zeros (or e_proj-only).

## File: src/models/deepseek4.cpp

### 5b: Add h_proj pathway alongside e_proj (~line 943)

Replace the e_proj-only MTP input block with e_proj + h_proj:

```cpp
    // MTP fusion input pathway
    bool is_mtp = hparams.nextn_predict_layers > 0 &&
                  il >= n_layer - hparams.nextn_predict_layers &&
                  layer.mtp_e_proj;

    if (is_mtp) {
        ggml_tensor * e_hc = nullptr;
        {
            // e_proj pathway: embed -> enorm -> e_proj -> repeat to HC
            ggml_tensor * e_normed = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            ggml_tensor * e_out = ggml_mul_mat(ctx0, layer.mtp_e_proj, e_normed);
            ggml_tensor * e_2d = ggml_reshape_3d(ctx0, e_out, n_embd, 1, n_tokens);
            e_hc = ggml_repeat_4d(ctx0, e_2d, n_embd, n_hc, n_tokens, 1);
            e_hc = ggml_reshape_3d(ctx0, e_hc, n_embd, n_hc, n_tokens);
        }

        ggml_tensor * h_hc = nullptr;
        if (layer.mtp_h_proj && session) {
            // h_proj pathway: prev_hc (from session) -> hnorm -> h_proj -> repeat to HC
            // session->mtp_prev_hc is stored after the output head in step 9
            ggml_tensor * prev_hc = ggml_view_3d(ctx0, session->mtp_prev_hc,
                n_embd, n_hc, n_tokens, 0, 0, 0);
            ggml_tensor * h_normed = ggml_rms_norm(ctx0, prev_hc, norm_rms_eps);
            ggml_tensor * h_out = ggml_mul_mat(ctx0, layer.mtp_h_proj, h_normed);
            ggml_tensor * h_3d = ggml_reshape_3d(ctx0, h_out, n_embd, n_hc, n_tokens);
            h_hc = h_3d;
        }

        if (h_hc) {
            inpL = ggml_add(ctx0, e_hc, h_hc);
        } else {
            inpL = e_hc;
        }
    } else {
        // Standard: repeat token embed to HC
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    }
```

### 5c: Store after_ffn_hc for next draft step (~line 1400, after output)

After the output head produces `cur`:

```cpp
    // Store hidden state for next MTP draft step
    if (is_mtp && session) {
        // session->mtp_prev_hc should be a persistent buffer allocated
        // in llama_new_context_with_model (size: n_embd * n_hc * max_batch)
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, inpL, session->mtp_prev_hc));
    }
```

### New struct field: src/llama-context.h or include/llama.h

Add to llama_context or session struct:

```cpp
    // MTP draft hidden state buffer (n_embd * n_hc * max_batch floats)
    struct ggml_tensor * mtp_prev_hc = nullptr;
```

Allocated in llama_new_context_with_model(), size = n_embd * n_hc * max_batch * sizeof(float).

## What this achieves

- First decode: prev_hc = zeros -> h_proj = zeros -> draft uses e_proj only
  (same as current behavior)
- Second+ decode: prev_hc = draft's own after_ffn_hc -> h_proj provides
  context from the draft's previous computation
- The draft model still doesn't see the target's hidden state, but it sees
  its OWN history, which is better than nothing

## Expected effect

- Short prompts (1-2 draft tokens): no change (first token is zeros-based)
- Long prompts (19 draft tokens): the recursive draft quality should improve
  because tokens 3-19 use the draft's own context
- On open-ended prompts at temperature 0.7: acceptance should improve from
  ~50% to ~60-70% (not as good as target-hidden-state's ~70-85%, but better
  than blind)

## Implementation effort

~30 lines added to deepseek4.cpp, ~5 lines to context init.
No new API, no cross-context communication.
Can be included in the next patch revision.
