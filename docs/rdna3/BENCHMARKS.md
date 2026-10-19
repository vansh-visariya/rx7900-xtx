# Benchmarks

## Test system

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, 24 GB (gfx1100), power limit 291 W |
| Driver | Mesa 26.0.8 RADV (Vulkan 1.4), Linux 7.0 |
| Host | Ryzen 3 3250U laptop (2 cores / 4 threads), 9 GB RAM, model files on an HDD |
| Link | eGPU over PCIe 3.0 x4 |

This is a weak host. The GPU does almost all of the work, but host-side gaps and PCIe copies cost more here than on a
desktop. On a desktop with a faster CPU and a PCIe 4.0 x16 slot, expect the same or better numbers.

## Method

`docs/rdna3/depth_bench.py` sends one request per depth to a running `llama-server`:

- a chat prompt of about N tokens made of real C/C++ source files of this repository (read with git at a fixed commit,
  so every build gets the same prompt), with a unique first line so that no prompt cache is reused,
- a coding question at the end, thinking on, temperature 1.0, top-p 0.95, top-k 20, a fixed seed, 1024 new tokens,
- it reports the server's own timings: prompt tokens per second (prefill) and generated tokens per second (decode).

The first request after a server start compiles shaders and is slower, so each run starts with a 4K warm-up request.
Prefill is a cold prefill of the whole prompt. Decode includes the draft acceptance rate, which depends on the text, so
it moves a few percent between prompts and seeds.

## Server commands

Model files: `Qwen3.8-27B-UD-Q4_K_M.gguf` (target, 16.5 GB) and `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (DFlash2 draft, 1.1 GB).
Both builds use the same flags where upstream has them. The two flags upstream does not have
(`--spec-accept rejection`, `--spec-draft-vocab`) are left out of the upstream command.

This fork:

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf -lm dio --device Vulkan0 -ngl 999 -fa on -c 220000 \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 1024 --ctx-checkpoints 3 -b 512 -ub 512 \
  --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-device Vulkan0 --spec-draft-ngl 999 \
  --spec-draft-n-max 4 --spec-accept rejection -bs --spec-draft-vocab 65536 \
  --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Upstream llama.cpp:

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf -lm dio --device Vulkan0 -ngl 999 -fa on -c 196608 \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 1024 --ctx-checkpoints 3 -b 512 -ub 512 \
  --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-device Vulkan0 --spec-draft-ngl 999 \
  --spec-draft-n-max 4 -bs \
  --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Then:

```bash
python3 docs/rdna3/depth_bench.py --port 8080 --depths 4000,16000,32000,48000,64000,128000,180000,200000 \
  --n-predict 1024 --out run.json
```

## Results (2026-10-19)

Builds: this fork at `8a0b44bdd` (branch `spec-90`), upstream llama.cpp master at `f174553e1`.

| Prefill, tokens/s | 16K | 32K | 48K | 64K | 128K | 180K | 200K |
|---|---:|---:|---:|---:|---:|---:|---:|
| upstream, no draft | 808.4 | 729.4 | 662.1 | 605.0 | 442.1 | 366.7 | not run |
| upstream + DFlash2 | 656.5 | 606.7 | 560.7 | GPU hang | GPU hang | GPU hang | no fit |
| this fork + DFlash2 | **871.8** | **850.3** | **816.7** | **780.5** | **656.6** | **583.5** | **558.5** |

| Decode, tokens/s | 16K | 32K | 48K | 64K | 128K | 180K | 200K |
|---|---:|---:|---:|---:|---:|---:|---:|
| upstream, no draft | 35.2 | 33.7 | 32.2 | 30.8 | 26.3 | 23.6 | not run |
| upstream + DFlash2 | 48.8 | 46.2 | 36.8 | GPU hang | GPU hang | GPU hang | no fit |
| this fork + DFlash2 | **87.4** | **88.8** | **93.2** | **89.8** | **69.1** | **69.8** | **66.2** |

Draft acceptance rate (accepted / drafted tokens), this fork: 0.43 / 0.46 / 0.51 / 0.51 / 0.40 / 0.46 / 0.45 at 16K / 32K /
48K / 64K / 128K / 180K / 200K; upstream + DFlash2: 0.39 / 0.43 / 0.35 at 16K / 32K / 48K. Part of the difference is
`--spec-accept rejection`, which upstream does not have. The decode dip at 128K follows the lower acceptance on that
prompt. The raw numbers are in [results/](results/).

### Context size

The largest `-c` each build fits on the 24 GB card with this model and the DFlash2 draft (no vision projector), and what
happens past it:

| Build | `-c` | VRAM in use at the first request | VRAM peak in the run (of 24560 MiB) |
|---|---|---|---|
| this fork | 220000 | 23934 MiB | 24086 MiB |
| upstream master | 196608 | 24049 MiB | 24323 MiB |
| upstream `f95b0d953` | 220000 | 24311 MiB | 24547 MiB: full, buffers moved to system RAM, decode 32 instead of 45 tokens/s at 4K |

The fork needs less VRAM mainly because the gated delta net layers roll back by replaying logged updates instead of
keeping state snapshots for every drafted token.

### The upstream DFlash2 hang

With DFlash2, upstream stopped every time in the 64K request, after about 60K prompt tokens:

```
E srv  update_slots: decode() failed: vk::Queue::submit: ErrorDeviceLost
kernel: amdgpu 0000:03:00.0: ring comp_1.1.0 timeout, signaled seq=..., emitted seq=...
kernel: amdgpu 0000:03:00.0: Ring comp_1.1.0 reset succeeded
```

- Upstream `f95b0d953` (the commit this fork branches from): stopped after 59904 prompt tokens, at `-c 220000` and at
  `-c 196608`.
- Upstream master `f174553e1`: stopped after 61440 prompt tokens at `-c 196608`.
- Upstream master without a draft: ran fine to 180K.
- This fork with DFlash2: ran fine to 200K, in two separate runs.

The GPU recovered through a queue reset each time. The cause was not investigated further.
