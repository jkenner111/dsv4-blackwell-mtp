#!/usr/bin/env python3
"""Apply MTP fusion wiring to deepseek4.cpp, then regenerate patch."""
base = '/tmp/fringe-analysis'

with open(f'{base}/src/models/deepseek4.cpp') as f:
    content = f.read()

# The standard inpL reshape/repeat block to replace
old = '''    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));'''

new = '''    // MTP fusion input pathway: embed -> enorm -> e_proj -> repeat to HC
    // h_proj pathway deferred (requires prev hidden state from session)
    if (hparams.nextn_predict_layers > 0 &&
        il >= n_layer - hparams.nextn_predict_layers &&
        layer.mtp_e_proj)
    {
        ggml_tensor * e_normed = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
        ggml_tensor * e_out = ggml_mul_mat(ctx0, layer.mtp_e_proj, e_normed);
        inpL = ggml_reshape_3d(ctx0, e_out, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    }
    else
    {
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    }

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));'''

assert old in content, 'Pattern not found in deepseek4.cpp'
with open(f'{base}/src/models/deepseek4.cpp', 'w') as f:
    f.write(content.replace(old, new))
print('OK fusion wiring applied')
