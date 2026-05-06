# arc-vllm

Run [vLLM](https://github.com/vllm-project/vllm) on an Intel Arc GPU
(Battlemage / Alchemist) and serve [Gemma 4](https://blog.google/technology/developers/gemma-4/)
through an OpenAI-compatible API.

## Why this exists

As of May 2026, the seam between "Gemma 4 supported" and "Intel XPU
supported" isn't covered by any published Docker image:

| Image                              | vLLM    | Gemma 4 | XPU |
| ---------------------------------- | ------- | ------- | --- |
| `intel/vllm:0.17.0-xpu`            | 0.17    | ❌ (pre-0.19) | ✅ |
| `intel/llm-scaler-vllm:0.14.0-b8`  | 0.14    | ❌      | ✅ (curated) |
| `vllm/vllm-openai:gemma4`          | 0.19    | ✅      | ❌ (CUDA only) |

So we build vLLM from source against `docker/Dockerfile.xpu` (pinned to a
known-good commit) and publish the resulting image to GHCR.

## Quick start

```bash
git clone https://github.com/nkhoit/arc-vllm.git
cd arc-vllm
docker compose up -d
docker compose logs -f vllm-gemma   # ~5 min cold start (load + warmup)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/gemma-4-E4B-it-AWQ-INT4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

That's it. The compose file pulls `ghcr.io/nkhoit/arc-vllm:latest`, mounts
`/dev/dri`, and serves an OpenAI-compatible API on port 8000.

## Hardware tested

- Intel **Arc Pro B50** (Battlemage G21, 16 GB) on Linux 6.x with the `xe`
  driver and level-zero
- Should also work on Arc Pro B60 (24/48 GB), Arc B580, and Alchemist (A770
  etc.) — adjust `MAX_MODEL_LEN` and `GPU_MEM_UTIL` per VRAM

## Configuration

Copy `.env.example` to `.env` and edit. All values are read by
`docker-compose.yml`:

| Var               | Default                                | Notes |
| ----------------- | -------------------------------------- | ----- |
| `MODEL`           | `cyankiwi/gemma-4-E4B-it-AWQ-INT4`     | Must use `compressed-tensors` quantization on XPU. See gotchas below. |
| `MAX_MODEL_LEN`   | `32768`                                | 8k–128k depending on free VRAM |
| `GPU_MEM_UTIL`    | `0.88`                                 | Fraction of VRAM vLLM may use |
| `HF_TOKEN`        | _(empty)_                              | Required for gated models |
| `HF_CACHE`        | `./hfcache`                            | Bind-mount for HuggingFace cache |
| `IMAGE`           | `ghcr.io/nkhoit/arc-vllm:latest`       | Set to `local` to build from source |

### VRAM budget (Gemma 4 E4B, INT4 weights)

| `MAX_MODEL_LEN` | Weights | KV cache | Headroom on 16 GB |
| ---------------:| ------- | -------- | ----------------- |
|             8k  | 9.9 GB  | ~0.2 GB  | comfortable, batch ok |
|            32k  | 9.9 GB  | ~0.9 GB  | single-stream + small batch |
|           128k  | 9.9 GB  | ~3.7 GB  | single-stream only, no headroom |

## Building from source

If you want to bump the pinned vLLM commit or build locally:

```bash
./setup.sh                     # clones vllm at the pinned commit
IMAGE=local docker compose up -d --build
```

Edit `.env.example` (or `.env`) to change `VLLM_COMMIT`.

## Gotchas

### Quantization on XPU

Per the [Intel Quantization Roadmap RFC](https://github.com/vllm-project/vllm/issues/37979),
the **only 4-bit scheme with merged XPU kernels right now is
compressed-tensors W4A16 INT** (a.k.a. `pack-quantized`). Watch out:

- A repo *named* `…-AWQ-…` may actually be compressed-tensors under the hood.
  Always check `quantization_config.quant_method` in `config.json`:
  - ✅ `compressed-tensors` → works
  - ❌ `awq` → crashes with `'_OpNamespace' '_C' object has no attribute 'awq_dequantize'`
- GPTQ, bnb-4bit, and FP8 are also unmerged on XPU as of vLLM 0.20.

Known-good 4-bit Gemma 4 builds:
- `cyankiwi/gemma-4-E4B-it-AWQ-INT4` (default)
- `cyankiwi/gemma-4-E4B-it-AWQ-INT8`

### `--enforce-eager` and `TRITON_ATTN`

XPU graphs and the default attention backend aren't fully stable on
Battlemage yet. Eager mode + Triton attention is the boring-but-working
combination.

### First request is slow

Cold Triton kernel compilation adds ~60 s to the first inference. Subsequent
requests are normal speed.

### Image size

The XPU image is ~28 GB compressed (bundles oneAPI 2025.3). Plan storage and
network accordingly.

## Why not just use `intel/vllm`?

It only ships up to vLLM 0.17, and Gemma 4 support landed in 0.19. As soon as
Intel publishes a 0.19+ XPU image with merged kernels, this repo can drop the
custom build and just consume that.

## License

MIT. vLLM, Gemma, and Intel oneAPI keep their own.
