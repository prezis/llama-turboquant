# llama-turboquant

Fork of [animehacker/llama-turboquant](https://github.com/animehacker/llama-turboquant) with a rope array length fix for RTX 5090 (SM_120 / Blackwell).

## What is TurboQuant / TQ3_0?

TQ3_0 implements Stage 1 (PolarQuant) of the [TurboQuant](https://arxiv.org/abs/2504.19874) pipeline for KV cache compression in [llama.cpp](https://github.com/ggml-org/llama.cpp):

- **3.5 bits per value** (14 bytes per block of 32 values)
- **4.57x compression** vs FP16 KV cache
- Walsh-Hadamard rotation + 3-bit Lloyd-Max codebook quantization
- Near-lossless quality (~5% perplexity increase on wikitext-2)

## This Fork's Change

The upstream code throws when a GGUF metadata array has fewer elements than expected (e.g., `rope_freq_base` arrays). Some models store shorter arrays that should be zero-filled rather than rejected.

**Fix in `src/llama-model-loader.cpp`:**
- Changed strict equality check to allow shorter arrays
- Zero-fills the result buffer, then reads available elements
- Prevents crash on model load with RTX 5090 / CUDA 12.8

## Build (NVIDIA / CUDA)

Requires CUDA 12.8+ and CMake 3.21+. Using micromamba:

```bash
micromamba create -n turboquant -c conda-forge cmake ninja
micromamba activate turboquant

git clone https://github.com/animehacker/llama-turboquant.git
cd llama-turboquant
mkdir build && cd build

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120  # RTX 5090

make -j$(nproc)
```

## Usage

```bash
# Serve Qwen3.5 with TQ3_0 KV cache compression
./bin/llama-server \
  -m model.gguf \
  --cache-type-k tq3_0 \
  --cache-type-v q8_0 \
  --reasoning-budget 0 \
  -ngl 99

# Benchmark
./bin/llama-bench -m model.gguf -ctk f16,tq3_0 -p 512,4096 -n 128,512
```

Use `--reasoning-budget 0` for Qwen3.5 to disable internal chain-of-thought and get raw generation speed.

## Benchmarks (RTX 5090, 32GB VRAM)

| Metric | Value |
|--------|-------|
| Max context (TQ3_0 K + Q8_0 V) | 26,733 tokens |
| Generation throughput | 188 tok/s |
| KV compression ratio | 4.57x vs FP16 |
| Perplexity impact | ~5% increase |

## References

- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni. *Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant** — *Quantizing KV Caches with Polar Transformation.* AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **Upstream fork** — [animehacker/llama-turboquant](https://github.com/animehacker/llama-turboquant) (V cache, flash attention, normalization fix)
- **Original implementation** — [unixsysdev/llama-turboquant](https://github.com/unixsysdev/llama-turboquant) (CUDA MMVQ kernel, block layout)

## License

MIT (same as llama.cpp)
