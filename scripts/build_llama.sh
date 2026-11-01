#!/usr/bin/env bash
# Build llama.cpp at the pinned commit with the flags this study depends on.
#
# The load-bearing flag is GGML_CUDA_FA_QUANTS: the three KV arms are exactly q4_0/q4_0,
# q8_0/q8_0 and f16/f16, and a rebuild that omits them changes both memory and numerics for the
# quantized-V arms (a quantized V cache requires flash attention). Everything else is ordinary.
#
# On hardware other than the study host, change CMAKE_CUDA_ARCHITECTURES only.
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-$HOME/Projects/code/clones/llama.cpp}"
COMMIT="${LLAMA_COMMIT:-2a667032dc788c3a071d538e0b572e45a3f12731}"
ARCHS="${CMAKE_CUDA_ARCHITECTURES:-native}"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --all --tags
git -C "$LLAMA_DIR" checkout --detach "$COMMIT"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" --target llama-server llama-tokenize llama-bench

echo
"$LLAMA_DIR/build/bin/llama-server" --version | head -3
echo "verify the FA quant list above appears in the build log:"
echo "  grep -r GGML_CUDA_FA_QUANTS $LLAMA_DIR/build/CMakeCache.txt"
