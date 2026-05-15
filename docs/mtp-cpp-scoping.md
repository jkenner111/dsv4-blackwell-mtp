# MTP Head C++ Changes — Line-Level Scoping

Adds 8 MTP-specific tensor slots to Fringe210's LLM_ARCH_DEEPSEEK4 so the
converted MTP GGUF (with e_proj, h_proj, hc_head_*, etc.) loads and the
graph builder runs the embedding/hidden fusion + MTP output head.

Existing infrastructure: Fringe210 already has `llama_layer_nextn` struct
(include/llama-model.h:204) and `LLM_TENSOR_NEXTN_*` enums (llama-arch.h:581)
used by other architectures. DeepSeek4 doesn't hook into them yet. This plan
adds new `LLM_TENSOR_MTP_*` enums specific to deepseek4 MTP mechanics.

---

## File 1: src/llama-arch.h

**Location:** After line 579 (LLM_TENSOR_HC_FFN_SCALE), before line 580 (LLM_TENSOR_FFN_GATE_TID2EID)

**Add 8 new enum entries:**

```cpp
    // MTP (Multi-Token Prediction) head tensors — DeepSeek V4
    LLM_TENSOR_MTP_E_PROJ,
    LLM_TENSOR_MTP_H_PROJ,
    LLM_TENSOR_MTP_ENORM,
    LLM_TENSOR_MTP_HNORM,
    LLM_TENSOR_MTP_HC_HEAD_BASE,
    LLM_TENSOR_MTP_HC_HEAD_FN,
    LLM_TENSOR_MTP_HC_HEAD_SCALE,
    LLM_TENSOR_MTP_NORM,
```

These are per-block tensors (layer_repeating), gated on `nextn_predict_layers > 0`.

---

## File 2: src/llama-arch.cpp

### 2a: Tensor name strings (~line 579, after HC_FFN_SCALE)

```cpp
    { LLM_TENSOR_MTP_E_PROJ,          "blk.%d.e_proj"           },
    { LLM_TENSOR_MTP_H_PROJ,          "blk.%d.h_proj"           },
    { LLM_TENSOR_MTP_ENORM,           "blk.%d.enorm"            },
    { LLM_TENSOR_MTP_HNORM,           "blk.%d.hnorm"            },
    { LLM_TENSOR_MTP_HC_HEAD_BASE,    "blk.%d.hc_head_base"     },
    { LLM_TENSOR_MTP_HC_HEAD_FN,      "blk.%d.hc_head_fn"       },
    { LLM_TENSOR_MTP_HC_HEAD_SCALE,   "blk.%d.hc_head_scale"    },
    { LLM_TENSOR_MTP_NORM,            "blk.%d.norm"              },
```

### 2b: Tensor info (layer/op) (~line 807, after HC_FFN_SCALE entry)

```cpp
    {LLM_TENSOR_MTP_E_PROJ,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_MTP_H_PROJ,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_MTP_ENORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_RMS_NORM}},
    {LLM_TENSOR_MTP_HNORM,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_RMS_NORM}},
    {LLM_TENSOR_MTP_HC_HEAD_BASE,     {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_ADD}},
    {LLM_TENSOR_MTP_HC_HEAD_FN,       {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_MTP_HC_HEAD_SCALE,    {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_SCALE}},
    {LLM_TENSOR_MTP_NORM,             {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_RMS_NORM}},
```

Notes:
- e_proj, h_proj, enorm, hnorm are per-layer (LLM_TENSOR_LAYER_REPEATING) — they sit inside the MTP block
- hc_head_* and norm are output-layer (LLM_TENSOR_LAYER_OUTPUT) — they produce the final embedding before lm_head

---

## File 3: include/llama-model.h (or wherever llama_layer is defined)

**Location:** struct llama_layer (~line 496), after the existing HC fields

**Add 8 new fields to llama_layer:**

```cpp
    // MTP (Multi-Token Prediction) head — DeepSeek V4
    struct ggml_tensor * mtp_e_proj         = nullptr;
    struct ggml_tensor * mtp_h_proj         = nullptr;
    struct ggml_tensor * mtp_enorm          = nullptr;
    struct ggml_tensor * mtp_hnorm          = nullptr;
    struct ggml_tensor * mtp_hc_head_base   = nullptr;
    struct ggml_tensor * mtp_hc_head_fn     = nullptr;
    struct ggml_tensor * mtp_hc_head_scale  = nullptr;
    struct ggml_tensor * mtp_norm           = nullptr;
```

These are nullable — only layers where `i >= n_layer - nextn_predict_layers` (MTP layers) will have them populated.

---

## File 4: src/llama-model.cpp

**Location:** LLM_ARCH_DEEPSEEK4 case, after the per-layer loop (~line 3290, before `} break;`)

**Inside the `for (int i = 0; i < n_layer; ++i)` loop, after the existing per-layer tensor creates, add:**

```cpp
                        // MTP head tensors for the last nextn_predict_layers
                        if (hparams.nextn_predict_layers > 0 &&
                            static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers)
                        {
                            layer.mtp_e_proj         = create_tensor(tn(LLM_TENSOR_MTP_E_PROJ,        "weight", i), {n_embd, n_embd}, 0);
                            layer.mtp_h_proj         = create_tensor(tn(LLM_TENSOR_MTP_H_PROJ,        "weight", i), {n_embd, n_embd}, 0);
                            layer.mtp_enorm          = create_tensor(tn(LLM_TENSOR_MTP_ENORM,         "weight", i), {n_embd}, 0);
                            layer.mtp_hnorm          = create_tensor(tn(LLM_TENSOR_MTP_HNORM,         "weight", i), {n_embd}, 0);
                            layer.mtp_hc_head_base   = create_tensor(tn(LLM_TENSOR_MTP_HC_HEAD_BASE,  "weight", i), {n_hc}, 0);
                            layer.mtp_hc_head_fn     = create_tensor(tn(LLM_TENSOR_MTP_HC_HEAD_FN,    "weight", i), {hc_dim, n_hc}, 0);
                            layer.mtp_hc_head_scale  = create_tensor(tn(LLM_TENSOR_MTP_HC_HEAD_SCALE, "weight", i), {1}, 0);
                            layer.mtp_norm           = create_tensor(tn(LLM_TENSOR_MTP_NORM,          "weight", i), {n_embd}, 0);
                        }
```

This gates creation on `nextn_predict_layers > 0` — the MTP GGUF sets this key. For a single-layer MTP draft with n_layer=1 and nextn_predict_layers=1, `i=0 >= 1-1=0` is true, so layer 0 gets the MTP tensors.

---

## File 5: src/models/deepseek4.cpp

This is where the forward pass needs branching. Two injection points:

### 5a: Input pathway — replace standard embedding with MTP fusion

**Location:** ~line 927, where `inpL = build_inp_embd(model.tok_embd)` currently runs.

For the MTP draft model running as a **standalone speculative draft** (not receiving hidden state from the target), we have two options:

**Option A (simple standalone):** The MTP draft embeds tokens normally, then runs the fusion pathway on its own hidden state from the previous step. This doesn't require target-model hidden-state injection — the draft is self-contained, just using the MTP-specific projection math.

```
// For MTP layers, replace inpL with fused input:
if (hparams.nextn_predict_layers > 0 && il >= n_layer - hparams.nextn_predict_layers) {
    // 1. Embed token normally
    ggml_tensor * embd = ggml_get_rows(ctx0, model.tok_embd, inp_tokens);
    // 2. RMSNorm embed (enorm)
    embd = ggml_rms_norm(ctx0, embd, norm_rms_eps);
    // 3. e_proj: embed -> HC dim
    ggml_tensor * e_out = ggml_mul_mat(ctx0, layer.mtp_e_proj, embd);
    ggml_tensor * e_hc  = ggml_repeat_4d(ctx0, e_out, n_embd, n_hc, n_tokens, 1);
    // 4. RMSNorm prev hidden state (hnorm)
    ggml_tensor * h_normed = ggml_rms_norm(ctx0, residual, norm_rms_eps);
    // 5. h_proj: hidden -> HC dim
    ggml_tensor * h_out = ggml_mul_mat(ctx0, layer.mtp_h_proj, h_normed);
    ggml_tensor * h_hc  = ggml_repeat_4d(ctx0, h_out, n_embd, n_hc, n_tokens, 1);
    // 6. Combine
    inpL = ggml_add(ctx0, e_hc, h_hc);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
}
```

Wait — `residual` is `inpL` at this point. We need to save the *previous* hidden state. Actually, for a standalone draft, the "previous hidden state" would be the `inpL` from the previous decode step, which isn't available inside a single graph build. This is a problem — the MTP draft in ds4 receives `cur_hc` from the *target* model, not from its own previous step.

For a standalone draft, the simplest correct behavior is: on the first step, use zero as the hidden input; on subsequent steps, use the output hidden state from the previous decode. This requires session-state management (storing `after_ffn_hc` between steps), which is a bigger change.

**Option B (deferred):** For the initial C++ port, just load and register the 8 tensors but don't wire the fusion pathway yet. The forward pass stays unchanged (standard 1-block deepseek4 trunk). The 8 tensors are present in the GGUF and loaded into `layer.mtp_*` fields, but unused in the graph. This is a step toward full MTP — it gets the loading pipeline correct, and the graph changes follow in a separate PR.

Given the scope, **Option B is the right first step.** Add the slots, load the tensors, verify the GGUF loads with all 38 tensors (30 trunk + 8 MTP), and the draft still works via the standard decode path. The fusion wiring (Option A) is a follow-up.

### 5b: Output pathway — use MTP output head instead of standard hc_head

**Location:** ~line 1378, where `dsv4_hc_head(ctx0, inpL, model.output_hc_fn, model.output_hc_scale, model.output_hc_base, ...)` runs.

For MTP layers, the output head uses the MTP-specific `hc_head_*` and `norm` weights instead of the global `output_hc_*`:

```cpp
// After the per-layer loop, before the output head:
if (hparams.nextn_predict_layers > 0) {
    // Use the MTP layer's own head weights
    const auto & last_layer = model.layers[n_layer - 1];
    if (last_layer.mtp_hc_head_fn) {
        cur = dsv4_hc_head(ctx0, inpL,
                last_layer.mtp_hc_head_fn,
                last_layer.mtp_hc_head_scale,
                last_layer.mtp_hc_head_base,
                n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
                norm_rms_eps, hparams.hc_eps);
    }
    if (last_layer.mtp_norm) {
        cur = build_norm(cur, last_layer.mtp_norm, nullptr, LLM_NORM_RMS, -1);
    }
} else {
    cur = dsv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
}
```

For Option B (load-only, no fusion), this output-head switch is simpler and can be included in the initial PR — it's a clean conditional that replaces two function calls with the MTP-layer-specific weights.

---

## convert_mtp_gguf.py update (P11)

Remove `MTP_DROP_SUFFIXES` — pass all 32 MTP tensors through with `mtp.0.* -> blk.0.*` renaming. The C++ loader now has slots for all 8 previously-dropped tensors. Expected total: 6 donor globals + 32 MTP remaps = 38 tensors.

Remove the `MTP_DROP_SUFFIXES` dict and the drop logic in the remap loop. That's a deletion of ~8 lines.

---

## Summary of edits

| File | Lines | Change |
|------|-------|--------|
| `src/llama-arch.h` | +8 | Add `LLM_TENSOR_MTP_*` enum entries |
| `src/llama-arch.cpp` | +24 | Add tensor name strings + tensor info entries |
| `include/llama-model.h` | +8 | Add `mtp_*` fields to `llama_layer` |
| `src/llama-model.cpp` | +12 | Gate `create_tensor` on `nextn_predict_layers` |
| `src/models/deepseek4.cpp` | +15 | Branch output head on `nextn_predict_layers` |
| `scripts/convert_mtp_gguf.py` | -8 | Remove MTP_DROP_SUFFIXES, pass through all 32 |

Total: ~75 lines added, ~8 removed. Option B (load-only) scope; Option A (fusion wiring) is follow-up.

---

## Verification after C++ changes

1. Convert MTP GGUF with updated script → output has 38 tensors (6 donor + 32 MTP)
2. Build Fringe210 with changes
3. Launch `llama-server --model-draft <converted.gguf>`
4. Expected: model loads without "wrong number of tensors" error
5. The draft still uses standard decode path (no fusion yet), so correctness is unchanged
6. The 8 MTP tensors are loaded into `layer.mtp_*` fields, ready for fusion wiring in follow-up
