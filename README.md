# dsv4-blackwell-mtp

MTP (Multi-Token Prediction) speculative decoding port for DeepSeek V4 Flash on NVIDIA Blackwell (GB10/DGX Spark).

## Goal

Make the -md (model-draft) flag in
[Fringe210/llama.cpp-deepseek-v4-flash-cuda](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda)
work with the MTP GGUF from
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
for speculative decoding on the DGX Spark.

## Status

- [x] Phase 1: Gap analysis
- [x] Phase 1: Quick-path GGUF conversion script
- [ ] Phase 1: Spark test with llama-server --model-draft
- [ ] Phase 2: Hidden-state injection for full ds4-level MTP performance
- [ ] Phase 3: PR to Fringe210

## Files

- docs/mtp-gap-analysis.md -- Full gap analysis: ds4 MTP vs Fringe210 CUDA
- docs/mtp-spark-test-plan.md -- Step-by-step test plan for DGX Spark
- scripts/convert_mtp_gguf.py -- Remaps mtp.0.* to blk.0.* tensors

## Quick start

```
python3 scripts/convert_mtp_gguf.py \
    DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
    DeepSeek-V4-MTP-deepseek4.gguf

llama-server \
    --model DeepSeek-V4-Flash-IQ2XXS-...gguf \
    --model-draft DeepSeek-V4-MTP-deepseek4.gguf \
    -ngl all -fa 1
```
