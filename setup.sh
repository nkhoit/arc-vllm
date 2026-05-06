#!/usr/bin/env bash
# Fetch the pinned vLLM source so docker compose can build the XPU image.
set -euo pipefail

VLLM_COMMIT="${VLLM_COMMIT:-3ca6ca2}"
VLLM_SRC="${VLLM_SRC:-./vllm}"

if [ -d "$VLLM_SRC/.git" ]; then
  echo "[arc-vllm] $VLLM_SRC already cloned. Checking out $VLLM_COMMIT…"
  git -C "$VLLM_SRC" fetch --depth 50 origin "$VLLM_COMMIT" 2>/dev/null || git -C "$VLLM_SRC" fetch origin
  git -C "$VLLM_SRC" checkout "$VLLM_COMMIT"
else
  echo "[arc-vllm] Cloning vLLM @ $VLLM_COMMIT into $VLLM_SRC…"
  git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC"
  git -C "$VLLM_SRC" checkout "$VLLM_COMMIT"
fi

echo "[arc-vllm] Ready. Run: docker compose up -d --build"
