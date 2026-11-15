# Windows and the AMD proprietary driver (RDNA4 report)

The fork was developed on Linux with Mesa RADV. This page covers what it takes to build and run it on Windows with the
AMD proprietary Vulkan driver, and what was measured on one RDNA4 card. Everything here comes from a single machine.

## Test system

| | |
|---|---|
| GPU | AMD Radeon RX 9070, 16 GB (RDNA4, detected by this fork as RDNA3) |
| Driver | AMD proprietary, Adrenalin 26.10.1 (Vulkan driver 2.0.416, LLPC) |
| Host | Ryzen 7 5800X, 32 GB RAM, Windows 11 |
| Compiler | MSVC 19.51 (Visual Studio 2026), Vulkan SDK 1.4.363 |

## Build

```
scripts\build-vulkan-windows.cmd
```

It needs Visual Studio with the C++ toolchain and the Vulkan SDK. Two portability fixes were needed for MSVC
(`clock_gettime` in the speculative trace and `__builtin_ctz` in the graph-replay key).

## What had to change for the proprietary driver

- **Grouped-query flash attention shader (`flash_attn_cm1_gqa.comp`).** The driver's shader compiler crashed with an
  access violation in `amdvlk64.dll` during `createComputePipeline`, so the server died at the first decode. Bisecting the
  shader showed that a store to an element of an array of cooperative matrices inside the KV loop is enough to trigger it.
  The S, P and O matrices are now separate variables.
- **Large matmul tile.** Port of upstream llama.cpp #31587: the large coopmat tile is enabled on the proprietary driver
  from 2.0.416 on. `GGML_VK_AMD_LARGE_TILE=0` keeps the old gate.
- **Split-K.** Ports of upstream llama.cpp #30280 and #30146. `GGML_VK_SPLIT_K_LARGE=0` and `GGML_VK_SPLIT_K_SMALLN=0`
  switch them off. No effect of #30146 on MTP decode was measured here, in either direction.
- **q4_0 KV cache.** The grouped-query shader had a fast load path for q8_0 K/V only; q4_0 went through the generic
  per-element decode and was slower than not using the shader at all (349 vs 502 tokens/s prefill at 32K). q4_0 now
  takes the same octet loads as q8_0 (18-byte blocks, low and high nibbles): 615 tokens/s.
- **Prompt batches.** On RADV the fork sends batches of a GQA model to the grouped-query shader instead of the f16
  scratch copy of K/V. On the proprietary driver the scratch copy is faster (32K prefill, no draft: 777 vs 615
  tokens/s), so batches of 64+ tokens keep it there by default and the shader takes the smaller ones.
  `GGML_VK_FA_GQA_PREFILL=1` forces the shader, `=0` disables it for all batch sizes.
- **bf16 K/V.** The driver also crashes compiling the coopmat1 flash attention shader with bf16 K/V; that case falls
  back to the scalar shader on the proprietary driver.

`GGML_VK_AMD_PROP_XL=1` opts in to the fork's RADV-only matmul work (128x256 tile, double-buffered tiles, f32
accumulators) on the proprietary driver. No effect was measured with `-ub 256`.

`GGML_VK_TRACE_PIPELINES=1` logs every pipeline before and after the driver compiles it, which is how the crashing
shader was found.

## Kernel tests

`test-backend-ops -b Vulkan0` on this card: `-o MUL_MAT` 1138/1138, `-o FLASH_ATTN_EXT` 5396/5396.

The non-GQA coopmat1 shader (`flash_attn_cm1.comp`) hit the same compiler crash for head size 128 and got the same
change: `SfMat` and `PVMat` are separate variables for the (at most two) row tiles.

## Results

Qwen3.8-27B Unsloth UD-Q2_K_XL (mostly IQ2 / IQ3 tensors), `-c 114688`, q4_0 KV cache, `-b 512 -ub 256`, flash attention,
vision projector on the CPU, temperature 1.0. Synthetic prompt, one pass per row, 256-512 generated tokens with
`ignore_eos`. Upstream is llama.cpp v1.0.0 (official Windows Vulkan build b12359, `b1b770d`, 2026-11-13); b12233 from
four days earlier measured the same within noise.

| Decode, tokens/s | short | 32K | 65K |
|---|---:|---:|---:|
| upstream, no draft | 35.4 | 33.4 | 31.6 |
| upstream + MTP 1 | 51.6 | 47.5 | 41.4 |
| upstream + MTP 3 | 36.5 | 43.1 | 39.5 |
| upstream + DFlash2 4 | 38.0 | 34.7 | 34.7 |
| this fork, no draft | 38.6 | 36.4 | 34.0 |
| this fork + MTP 1 | 72.3 | 61.6 | 55.5 |
| this fork + MTP 3, `--spec-accept rejection -bs --spec-draft-vocab 65536` | 84.8 | 66.1 | 53.8 |
| this fork + DFlash2 4, same flags | 71.6 | 62.3 | 52.2 |

| Prefill, tokens/s | 0-32K | 32K-65K |
|---|---:|---:|
| upstream, no draft | 761 | 501 |
| upstream + MTP 1 | 728 | 456 |
| upstream + DFlash2 4 | 744 | 478 |
| this fork, no draft | 795 | 509 |
| this fork + MTP 1 | 662 | 354 |
| this fork + MTP 3 | 644 | 358 |
| this fork + DFlash2 4 | 710 | 464 |

Decode with a draft is 26-64% faster than upstream + MTP 1. Prefill without a draft and with the DFlash2 draft is on
par with upstream; with an MTP draft it is still 9-22% slower. Peak VRAM is the same as upstream with MTP (13.6 GiB
against 13.7) and 0.6 GiB higher with the DFlash2 draft (14.2 GiB).
