# DeepSeek V4 Flash MTP speculative decoding for Blackwell CUDA

This repository ports the MTP (Multi-Token Prediction) speculative decoding from
[antirez/ds4](https://github.com/antirez/ds4) to the
[Fringe210 CUDA fork](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda)
of llama.cpp, targeting the NVIDIA DGX Spark (GB10, SM121 Blackwell).

Fringe210 already implements CUDA kernels for all five DSv4 ops: DSV4_ROPE_TAIL,
DSV4_HC_SPLIT_SINKHORN, DSV4_HC_EXPAND, DSV4_HC_WEIGHTED_SUM, DSV4_FP8_KV_QUANTIZE.

This project makes --model-draft work with the MTP GGUF from
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf).

## Status

- [x] Phase 1: Gap analysis
- [x] Phase 1: V1 conversion script (state-machine bugs fixed on Spark)
- [x] Phase 1: V2 script with --donor flag (merges token_embd/output from main model)
- [ ] Phase 1: Spark test v2 -- server starts with both models loaded
- [ ] Phase 2: Hidden-state injection for full ds4-level MTP performance
- [ ] Phase 3: PR to Fringe210

## Quick start

### 1. Convert the MTP GGUF (v2 with donor merge)

```bash
python3 scripts/convert_mtp_gguf.py \
    DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
    --donor DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
    DeepSeek-V4-MTP-deepseek4.gguf
```

The --donor flag copies token_embd.weight, output_norm.weight, and output.weight
from the main model. The MTP GGUF doesn't contain these -- ds4 references the base
model's weights at runtime. The converted output is a self-contained single-layer
deepseek4 GGUF.

### 2. Launch with speculative decoding

```bash
pkill -f llama-server && sleep 3
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

nohup ~/llama-dsv4-fringe/build/bin/llama-server \
    --model DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
    --model-draft DeepSeek-V4-MTP-deepseek4.gguf \
    --host 0.0.0.0 --port 8085 \
    --ctx-size 131072 -ngl all -fa 1 \
    --batch-size 4096 --threads 10 \
    --no-mmap --direct-io -np 1 --jinja \
    --alias deepseek-v4-flash-mtp \
    > /tmp/fringe-mtp-server.log 2>&1 &
```

### 3. Verify

```bash
sleep 10
tail -30 /tmp/fringe-mtp-server.log
# Should show both models loaded, then "server is listening"

curl -s http://localhost:8085/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash-mtp","messages":[{"role":"user","content":"Hi"}],"max_tokens":20,"temperature":0}'
```

## Files

| File | Purpose |
|------|---------|
| `scripts/convert_mtp_gguf.py` | V2: GGUF remapping with --donor merge |
| `docs/mtp-gap-analysis.md` | Full gap analysis: ds4 MTP vs Fringe210 CUDA |
| `docs/mtp-spark-test-plan.md` | Step-by-step DGX Spark test procedure |

## Related projects

- [Fringe210/llama.cpp-deepseek-v4-flash-cuda](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda) — CUDA fork running on the Spark
- [antirez/ds4](https://github.com/antirez/ds4) — Original Metal MTP implementation
- [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) — Model files on HF
