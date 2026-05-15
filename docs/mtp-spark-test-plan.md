# MTP Quick-Path Test Plan for DGX Spark (GB10)

## Prerequisites

The MTP GGUF file must be downloaded from HF:
  antirez/deepseek-v4-gguf :: DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf

Place it in:
  /home/borisdgx/models/deepseek-v4-flash/

## Step 1: Convert the MTP GGUF

Run on the Spark (where Fringe210 gguf-py is available):

```bash
cd ~/llama-dsv4-fringe
python3 gguf-py/gguf/scripts/convert_mtp_gguf.py \
  /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-MTP-deepseek4.gguf
```

Alternatively, use the standalone script at /tmp/convert_mtp_gguf.py:

```bash
python3 /tmp/convert_mtp_gguf.py \
  /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-MTP-deepseek4.gguf
```

Expected output:
- Lists all tensors with remapped names (mtp.0.* -> blk.0.*)
- Reports no errors
- Produces output file of similar size (~3.5 GB)

## Step 2: Launch with speculative decoding

```bash
pkill -f llama-server && sleep 3

nohup ~/llama-dsv4-fringe/build/bin/llama-server \
  --model /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --model-draft /home/borisdgx/models/deepseek-v4-flash/DeepSeek-V4-MTP-deepseek4.gguf \
  --host 0.0.0.0 --port 8085 \
  --ctx-size 131072 -ngl all -fa 1 \
  --batch-size 4096 --threads 10 \
  --no-mmap --direct-io -np 1 --jinja \
  --alias deepseek-v4-flash-fringe-mtp \
  > /tmp/fringe-mtp-server.log 2>&1 &
```

Key additions vs current command:
- `--model-draft` pointing to the converted MTP GGUF
- Different log file to separate from non-MTP runs

## Step 3: Verify

```bash
# Check server started
sleep 5
tail -20 /tmp/fringe-mtp-server.log

# Should see both models loaded without errors
# If it fails with "tensor not found", conversion didn't work
# If it fails with "architecture", the arch field needs adjustment

# Quick inference test
curl -s http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-fringe-mtp",
    "messages": [{"role": "user", "content": "Say hello in exactly 3 words"}],
    "max_tokens": 20,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

## Step 4: Benchmark

```bash
# Compare against current baseline (30.83 prompt t/s, 8.90 gen t/s)
# Run the same prompt with and without --model-draft
# Measure tokens/sec in the server log or via timing
```

## Expected Behavior

- **Best case:** MTP draft accepts ~50-70% of proposed tokens, yields 1.2-1.5x generation speedup
- **Worst case:** MTP loads but acceptance rate is low (draft model too different from target), minimal speedup
- **Failure mode 1:** "tensor not found: blk.0.attn_q_a.weight" -> conversion script missed some tensors
- **Failure mode 2:** "unsupported architecture" -> general.architecture not set to deepseek4
- **Failure mode 3:** Server crashes during generation -> likely KV cache incompatibility between draft/target

## Notes

- This is the "quick path" — the MTP runs as an independent draft model, not sharing hidden state with the target. It will be slower than ds4's native MTP but should still provide some speedup.
- The converted GGUF is only ~3.5 GB, easily fits alongside the 82 GB main model on the 128 GB Spark.
- If this works, the Phase 2 PR adds hidden-state injection for full ds4-level performance.
