# DeepSeek V4 Flash MTP speculative decoding for Blackwell CUDA

This repository ports the MTP (Multi-Token Prediction) speculative decoding from
[antirez/ds4](https://github.com/antirez/ds4) to the
[Fringe210 CUDA fork](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda)
of llama.cpp, targeting the NVIDIA DGX Spark (GB10, SM121 Blackwell).

The Fringe210 fork already implements CUDA kernels for all five DSv4 operations:

- `DSV4_ROPE_TAIL`
- `DSV4_HC_SPLIT_SINKHORN`
- `DSV4_HC_EXPAND`
- `DSV4_HC_WEIGHTED_SUM`
- `DSV4_FP8_KV_QUANTIZE`

This project makes the `--model-draft` (`-md`) flag in `llama-server` work with the MTP GGUF
(`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`) from [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf),
enabling speculative decoding on Blackwell GPUs.

## Status

- [x] Phase 1: Gap analysis — ds4 MTP (Metal) vs Fringe210 speculative decoding (CUDA)
- [x] Phase 1: Quick-path GGUF tensor remapping script
- [ ] Phase 1: Spark test — run converted MTP GGUF with `llama-server --model-draft`
- [ ] Phase 2: Hidden-state injection for full ds4-level MTP performance
- [ ] Phase 3: PR to Fringe210

## Quick start

### 1. Convert the MTP GGUF

The ds4 MTP GGUF uses `mtp.0.*` tensor names. Fringe210 expects `blk.0.*` for the
`deepseek4` architecture. Convert it:

```bash
python3 scripts/convert_mtp_gguf.py \
    DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
    DeepSeek-V4-MTP-deepseek4.gguf
```

### 2. Launch with speculative decoding

```bash
~/llama-dsv4-fringe/build/bin/llama-server \
    --model DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
    --model-draft DeepSeek-V4-MTP-deepseek4.gguf \
    --host 0.0.0.0 --port 8085 \
    --ctx-size 131072 -ngl all -fa 1 \
    --batch-size 4096 --threads 10 \
    --no-mmap --direct-io -np 1 --jinja \
    --alias deepseek-v4-flash-mtp
```

### 3. Verify

```bash
curl -s http://localhost:8085/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash-mtp","messages":[{"role":"user","content":"Hi"}],"max_tokens":20,"temperature":0}'
```

## Files

| File | Purpose |
|------|---------|
| `scripts/convert_mtp_gguf.py` | GGUF tensor remapping (mtp.0.* -> blk.0.*) |
| `docs/mtp-gap-analysis.md` | Full gap analysis: ds4 MTP vs Fringe210 CUDA |
| `docs/mtp-spark-test-plan.md` | Step-by-step DGX Spark test procedure |

## Related projects

- [Fringe210/llama.cpp-deepseek-v4-flash-cuda](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda) — CUDA fork running on the Spark
- [antirez/ds4](https://github.com/antirez/ds4) — Original Metal MTP implementation
- [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) — Model files on Hugging Face
