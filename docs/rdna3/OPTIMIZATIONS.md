# What this fork changes

This fork branches from upstream llama.cpp at `f95b0d953` (2026-09-22). All changes target the Vulkan backend on AMD
RDNA3 (developed and measured on a Radeon RX 7900 XTX, gfx1100, Mesa RADV) and the Qwen3.5 / Qwen3.8 hybrid architecture
(gated delta net + grouped-query attention) with speculative decoding (DFlash2 or MTP drafts).

Everything listed here is **on by default**. Each item has an environment variable that turns it off, so you can check
what it does on your own hardware. The environment variables only switch features off: you do not need any of them to
get the speed.

Most items are bit-exact: the same request gives the same text as before. The exceptions are the K-quant mat-vec
rounding change, the prompt-batch attention kernel and the f32 matmul accumulators. Each of these passed the
`test-backend-ops` suite and a 32-task quality eval with and without thinking.

## Decode (token generation)

| Change | What it does | Switch (off) |
|---|---|---|
| Integer-dot mat-vecs for IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S | packed int8 dot products, each sub-block unpacked once for all columns | `GGML_VK_DISABLE_MMVQ=1` |
| Q4_K / Q5_K unpack-once, Q6_K integer path for batches | same for K-quants | - |
| Grouped-query flash attention for verify batches (`flash_attn_cm1_gqa.comp`) | all tokens x heads of a GQA group in one workgroup, O kept in accumulators, q8_0 K/V loaded in 12-byte octets, 32-wide subgroups | `GGML_VK_FA_GQA_SHADER=0` |
| Graph fusions | residual add into 2-8 column mat-vecs; norm / swiglu write the q8_1 copy for the next integer mat-vec; RMS_NORM+MUL(+MUL), RMS_NORM+SCALE; bias / softplus / sigmoid / silu in the mat-vec epilogue | `GGML_VK_FUSE_MMV_ADD=0`, `GGML_VK_FUSE_Q8_1=0`, `GGML_VK_FUSE_MMV_EPI=0` |
| Gated delta net state in place | the delta net reads its state from the cache row; conv + conv-state path in one kernel | `GGML_VK_FUSE_GDN_STATE=0`, `GGML_VK_FUSE_CONV_STATE=0` |
| Delta-net rollback by replay | logs (k, delta, g) per token and replays the accepted steps in the next call, instead of writing 5 state snapshots per layer per verify. Uses about 600 MB less VRAM at long context | `LLAMA_GDN_REPLAY=0` |
| Command-buffer replay | a graph that repeats (verify, draft) is recorded once and resubmitted, matched by the scheduler's graph uid | `GGML_VK_GRAPH_REPLAY=0` |
| Host path | spin wait, early first submit, one backend wait per split, KV rotation matrices kept on the device, causal KQ mask built on the device | `GGML_VK_SPIN_WAIT=0`, `GGML_VK_FIRST_SUBMIT_NODES=0`, `LLAMA_KQ_MASK_GPU=0` |
| DFlash2 draft | layer features kept on the device, the block convolution as one `ggml_dflash_conv` op (was ~10 small nodes), deferred streaming | `LLAMA_DFLASH_DEV_INJECT=0`, `LLAMA_DFLASH_CONV_OP=0`, `LLAMA_SPEC_DEFER=0` |

## Prefill (prompt processing)

| Change | What it does | Switch (off) |
|---|---|---|
| 128x256 quant matmul tile | 4th tile size for batches of 256+ columns (use `-ub 512`) | - |
| f32 matmul accumulators on RDNA3 | an f16 WMMA accumulator costs the same registers on RDNA3, and the compiler moved f16 accumulators in 16-bit pieces | `GGML_VK_MM_F16ACC=1` |
| Double-buffered matmul tiles | the next K block is loaded into a second shared-memory buffer while the current one is multiplied | `GGML_VK_MM_DBUF=0` |
| Prompt-batch flash attention (`flash_attn_cm1_gqa_pp.comp`) | 64 query rows per workgroup, S and O transposed so the softmax runs per lane, K/V dequantized once per 32-row tile | `GGML_VK_FA_GQA_PP=0` |
| KV split for the last workgroup round | a 512-token batch filled 5 rounds on 96 CUs with an 86% empty last round | `GGML_VK_FA_GQA_TAIL=0` |
| `-lm dio` through a bounce buffer | direct IO used to fall back to buffered IO (O_DIRECT into pinned Vulkan memory fails), which pushed the whole model through the page cache. Matters on hosts with less RAM than the model | - |

## Speculative decoding and sampling (new flags)

| Flag | What it does |
|---|---|
| `--spec-accept rejection` | lossless rejection-sampling verification instead of exact match. Same output distribution, more accepted tokens at temperature > 0 |
| `-bs` (backend sampling) | now also works with the reasoning budget and with tool-call grammars, until the grammar starts constraining |
| `--spec-draft-vocab N` | the MTP draft head scores only the first N token ids. 65536 keeps the acceptance rate unchanged |

Note: the llama-server web UI sends `backend_sampling: false` unless you tick Settings -> Sampling -> "Backend
sampling". API clients that do not send the field get the server's `-bs` setting.

## Diagnostics

| Variable | Output (stderr) |
|---|---|
| `LLAMA_SPEC_TRACE=N` | per-cycle host timeline of the speculative loop (medians) |
| `LLAMA_SPEC_STATS=N` | draft / verify / accept times and acceptance estimates |
| `GGML_VK_GPU_SPAN=N` | GPU busy span and idle gap before each graph |
| `GGML_VK_CPU_TIMING=N` | host record / submit / sync time per graph size class, replay hits |
| `GGML_VK_REPLAY_WHY=N` | why graphs miss the replay cache |
| `GGML_VK_FUSE_ALIAS_LOG=1` | in-place fusions skipped because of allocator overlap |
| `GGML_SCHED_TIMING=N` | input copy and compute call time per scheduler split |
| `LLAMA_DECODE_TIMING=N` | llama_decode phases per context |

## Tests added

`test-backend-ops` has cases for the new ops and shapes (`GATED_DELTA_NET_REPLAY`, `DFLASH_CONV`, `SSM_CONV_STATE`,
`MUL_MAT_VEC_EPILOGUE`, `MUL_MAT_MODEL`, q8_0 prompt-batch and verify flash attention at this model's head sizes) and
perf cases at the model's real shapes. Run them on your card before trusting it:

```bash
build/bin/test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0
build/bin/test-backend-ops -o MUL_MAT -b Vulkan0
```
