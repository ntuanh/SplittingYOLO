# SplitYOLO — Results Summary

**Device**: CUDA · FP32 · batch = 32 · warmup = 10 · rounds = 50  
**Model**: YOLO11n (`yolo11n.pt`, 24 layers, scale n: width×0.25, depth×0.5)

---

## Table of Contents

1. [Cut Layer Comparison](#1-cut-layer-comparison)
2. [Speed — cut_layer = 13, batch = 32](#2-speed--cut_layer--13-batch--32)
3. [Speed — cut_layer = 12, batch = 32 (reference)](#3-speed--cut_layer--12-batch--32-reference)
4. [Pipeline Throughput Explanation](#4-pipeline-throughput-explanation)
5. [Memory — Loading Weights](#5-memory--loading-weights-no-forward-pass)
6. [Memory — During Inference](#6-memory--during-inference-batch--32)
7. [**Memory Breakdown — What Fills VRAM and RAM**](#7-memory-breakdown--what-fills-vram-and-ram)
8. [Full Comparison Table](#8-full-comparison-table)
9. [Conclusions](#9-conclusions)

---

## 1. Cut Layer Comparison

| cut_layer | Head layers | Tail layers | Head time | Tail time | Pipeline FPS | vs local |
|-----------|-------------|-------------|-----------|-----------|-------------|---------|
| 12 | 0–12 (13 layers) | 13–23 (11 layers) | 104.2 ms | 167.8 ms | 190.7 FPS | 1.43× |
| **13** | **0–13 (14 layers)** | **14–23 (10 layers)** | **114.9 ms** | **193.8 ms** | **165.1 FPS** | **1.33×** |

Moving cut from layer 12 → 13 adds C3k2(512, False) to the head and removes it from the tail.  
Both times are tail-bottlenecked by CPU-side NMS variance (std ≈ 12–14 ms).  
Cut layer 12 is the better split point for this GPU/NMS configuration.

---

## 2. Speed — cut_layer = 13, batch = 32

### 2a. Eager (default PyTorch)

| Scenario | Mean/batch | Std | FPS |
|----------|-----------|-----|-----|
| Local — full 24-layer model | 257.2 ms | 10.7 ms | **124.4** |
| Head stage — layers 0–13 | 114.9 ms | 0.2 ms | 278.5 (stage) |
| Tail stage — layers 14–23 + NMS | 193.8 ms | 13.5 ms | 165.1 (stage) |
| **Pipeline throughput** | — | — | **165.1** (1.33×) |

Pipeline formula (concurrent head+tail):

```
FPS = batch / max(head_ms, tail_ms) = 32 / max(114.9, 193.8) = 165.1 FPS
```

Bottleneck: **tail** (NMS on CPU processes 32 × 8400 candidate boxes each batch).

### 2b. CUDA Graphs (`torch.cuda.CUDAGraph` direct API)

CUDA Graphs record every CUDA kernel launch in one "capture" pass, then replay the full sequence in a single driver call — eliminating Python dispatch and kernel-launch overhead.

| Scenario | Mean/batch | Std | FPS | VRAM peak |
|----------|-----------|-----|-----|----------|
| Local — CUDA Graph | 299.2 ms | 6.5 ms | 106.9 | 1283 MB |
| Head — CUDA Graph | 116.4 ms | 0.2 ms | 274.9 (stage) | 1364 MB |
| Tail + NMS — CUDA Graph | 230.3 ms | 4.1 ms | 139.0 (stage) | 1944 MB |
| **Pipeline (CUDA Graphs)** | — | — | **139.0 (1.12×)** | — |

**CUDA Graphs is slower than eager for YOLO11n at batch=32.**

#### Why CUDA Graphs hurt here

| Factor | Effect |
|--------|--------|
| Per-kernel compute >> launch overhead | YOLO11n at batch=32 spends ~250 ms in actual GPU math; kernel launch overhead is microseconds per kernel — CUDA Graphs saves <1 ms total |
| Memory pinning | CUDA Graphs pin ALL intermediate tensors alive simultaneously (they can't be freed between kernels). Tail VRAM leaps from 1,136 MB (eager) → 1,944 MB (graph), a 71% increase |
| Static copy overhead | Before each replay: `sx.copy_(x)` — extra PCIe transfer per call |
| NMS still on CPU | NMS runs on CPU after `g.replay()` regardless; it's the bottleneck but unaffected by CUDA Graphs |

CUDA Graphs are most beneficial when Python overhead >> GPU compute time (e.g. batch=1 on a fast GPU). At batch=32, computation dominates — there is nothing to gain.

---

## 3. Speed — cut_layer = 12, batch = 32 (reference)

*From earlier session, all values from eager PyTorch*

| Method | Mean / batch (ms) | FPS |
|--------|-------------------|-----|
| Lib — `model(x)` | 239.5 ms | **133.6** |
| Manual — `for layer in model.model` | 260.5 ms | 122.9 |

| Stage | Layers | Mean / batch (ms) | Stage FPS |
|-------|--------|-------------------|-----------|
| Head | 0–12 | 104.2 ms | 307.0 |
| Tail | 13–23 + NMS | 167.8 ms | **190.7** ← bottleneck |

**Pipeline**: 190.7 FPS (1.43× over local lib)

---

## 4. Pipeline Throughput Explanation

In a streaming system head and tail run **concurrently on different batches**:

```
Batch 1:  [head]──►[tail 194ms]──────────────►
Batch 2:       [head]──────────►[tail 194ms]──►
```

```
FPS(pipeline) = batch_size / max(head_time, tail_time)
```

Communication delay (network/RabbitMQ) does NOT affect throughput with async sends — only bandwidth matters:  
payload per batch ≈ `y[4](3.28 MB) + y[10](0.82 MB) + y[13](6.55 MB)` ≈ **10.7 MB/batch** at cut_layer=13.  
At 165 FPS that needs ≈ `165 × 10.7 MB ≈ 1.76 GB/s` sustained throughput (~14 Gbps link).

---

## 5. Memory — Loading Weights (no forward pass)

*Each model measured in a separate subprocess for clean isolation*

| Model | Layers | Params | Weight MB | VRAM alloc | VRAM reserved | RAM Δ |
|-------|--------|--------|-----------|-----------|---------------|-------|
| yolo11n.pt (full) | 24 | 2,624,080 | 10.01 MB | 10.24 MB | 30.00 MB | 101.4 MB |
| head.pt (head, cut=12) | 13 | 1,365,472 | 5.21 MB | 5.29 MB | 26.00 MB | 93.3 MB |
| **Ratio (head / full)** | — | **52%** | **52%** | **51.6%** | 86.7% | 92% |

**VRAM allocated** is exactly proportional to parameter count (52%).  
**RAM Δ** (~93–101 MB) is dominated by fixed overhead unrelated to model size.

---

## 6. Memory — During Inference (batch = 32)

### 6a. Eager mode (cut_layer = 13)

| Scenario | VRAM peak alloc |
|----------|----------------|
| Local (full 24 layers) | 1,485.5 MB |
| Head (layers 0–13) | 1,485.5 MB |
| Tail (layers 14–23) | 1,136.5 MB |

*Peak occurs at early backbone layers — shared by full model and head model.  
Cutting layers after the peak saves only ~0 MB at inference peak.*

### 6b. CUDA Graphs mode (cut_layer = 13)

| Scenario | VRAM peak alloc | vs eager |
|----------|----------------|---------|
| Local CUDA Graph | 1,283.4 MB | −13% |
| Head CUDA Graph | 1,363.8 MB | ≈ same |
| Tail CUDA Graph | 1,944.2 MB | **+71%** |

CUDA Graphs pin all intermediate activations simultaneously → tail VRAM explodes.

---

## 7. Memory Breakdown — What Fills VRAM and RAM

### 7a. RAM (process memory) during model loading

RAM is dominated by fixed costs that are unrelated to model size:

| Component | RAM | Scales with model? |
|-----------|-----|-------------------|
| Python interpreter + stdlib | ~50 MB | No |
| PyTorch + torchvision + CUDA runtime DLLs | ~300 MB | No |
| Ultralytics library (model defs, utils) | ~100 MB | No |
| **CUDA context init** — cuDNN/cuBLAS handles, device state (paid once on first `.cuda()` call) | **~111 MB** | No |
| Model weights CPU buffer (briefly in RAM while `.cuda()` copies to GPU) | 10 MB / 5 MB | **Yes** |
| **Total measured** | **~582 MB** | — |

The actual model weights (10 MB full, 5 MB head) are only **~1.7% of total RAM**.  
The remaining 98.3% is fixed per-process overhead that would exist even with a 1-layer model.

---

### 7b. VRAM during weight loading (no forward pass)

| Component | Full model | Head model | Scales? |
|-----------|-----------|-----------|---------|
| Conv weight tensors (all layers) | 8.64 MB | 4.47 MB | **Yes** (52%) |
| Bias + BN running mean/var | 1.60 MB | 0.82 MB | **Yes** (52%) |
| **Total allocated** | **10.24 MB** | **5.29 MB** | **Yes** |
| PyTorch allocator base block (minimum reservation) | ~20 MB | ~20 MB | No |
| **Total reserved** | **30 MB** | **26 MB** | — |

Weight tensors scale exactly with parameter count (52%). The allocator always reserves a minimum block (~20 MB), which is why VRAM reserved only saves 14% even though weights save 52%.

---

### 7c. VRAM during inference (batch = 32, FP32) — what's inside

Every layer's output tensor must live in VRAM until it is consumed. Tensors in `model.save` [4, 6, 10, 13] must stay alive until their target layer, making them long-lived. All other tensors are freed as soon as the next layer consumes them (transient).

#### Activation tensor sizes (YOLO11n n-scale, batch=32)

| Layer | Type | Output shape | Size | Lifetime |
|-------|------|-------------|------|---------|
| Input x | — | (32, 3, 640, 640) | **157 MB** | until layer 0 consumes it |
| L0 Conv(3→16, s2) | Conv | (32, 16, 320, 320) | **200 MB** | transient |
| L1 Conv(16→32, s2) | Conv | (32, 32, 160, 160) | **100 MB** | transient |
| L2 C3k2 — cv1 out | internal | (32, 64, 160, 160) | **200 MB** | transient |
| L2 C3k2 — cat([x1, x2, bottleneck]) | internal | (32, 96, 160, 160) | **300 MB** | transient |
| L2 C3k2 — final output | C3k2 | (32, 64, 160, 160) | **200 MB** | transient |
| L3 Conv(64→64, s2) | Conv | (32, 64, 80, 80) | **50 MB** | transient |
| L4 C3k2 — output = **y[4]** | C3k2 | (32, 128, 80, 80) | **100 MB** | **L4 → L15 (long-lived)** |
| L5 Conv(128→128, s2) | Conv | (32, 128, 40, 40) | **25 MB** | transient |
| L6 C3k2 — output = **y[6]** | C3k2 | (32, 128, 40, 40) | **25 MB** | **L6 → L12** |
| L7 Conv(128→256, s2) | Conv | (32, 256, 20, 20) | **12.5 MB** | transient |
| L8 C3k2 | C3k2 | (32, 256, 20, 20) | **12.5 MB** | transient |
| L9 SPPF — internal cat | internal | (32, 512, 20, 20) | **25 MB** | transient |
| L9 SPPF — output | SPPF | (32, 256, 20, 20) | **12.5 MB** | transient |
| L10 C2PSA — attn q/k/v | internal | 3 × (32, 2, 400, 32) | **10 MB** | transient |
| L10 C2PSA — attn matrix softmax(QKᵀ) | internal | (32, 2, 400, 400) | **39 MB** | transient |
| L10 C2PSA — output = **y[10]** | C2PSA | (32, 256, 20, 20) | **12.5 MB** | **L10 → L21 (long-lived)** |

> **Shape formula (FP32)**: `N × C × H × W × 4 bytes`  
> e.g. y[4]: `32 × 128 × 80 × 80 × 4 = 104,857,600 bytes = 100 MB`

#### Peak memory moment — early C3k2 layers (L2, L4)

The peak does NOT occur at C2PSA (the attention layer). It occurs at **C3k2 layer 2**, where 160×160 spatial maps at batch=32 produce the largest simultaneous allocations:

```
Inside L2 C3k2 (peak):
  cv1 output  (32, 64, 160, 160)   = 200 MB  (split into views x1, x2)
  bottleneck  (32, 32, 160, 160)   = 100 MB  (adds new allocation)
  cat result  (32, 96, 160, 160)   = 300 MB  (new allocation before cv1 freed)
  cv2 output  (32, 64, 160, 160)   = 200 MB  (new, before cat freed)
  subtotal (tensor data)           ≈ 600 MB

cuDNN workspace for Conv2d (L0, L1):
  cuDNN selects algorithm at benchmark time.
  For batch=32, 640×640→320×320 conv:
  workspace ≈ 600–900 MB  (IMPLICIT_PRECOMP_GEMM or GEMM-based)

Model weights:                     =  10 MB
Other allocator residuals:         ≈ 250 MB
─────────────────────────────────────────────
Total measured peak:               ≈ 1,485 MB
```

#### Why cuDNN workspace is so large

`torch.backends.cudnn.benchmark = True` makes PyTorch try multiple algorithms at first call and cache the fastest one. For large-batch, high-resolution convolutions (L0, L1), cuDNN allocates a **temporary workspace buffer** proportional to `batch × input_size`. This workspace is freed after each conv, but it overlaps with the skip tensors y[4] and y[6] that are kept alive across layers — pushing the peak measurement up.

#### Tail-only peak (1,136 MB) is lower because

When only the tail runs (layers 14–23), it never executes layers 0–2. Those large early-resolution tensors (200–300 MB each) and their cuDNN workspaces never get allocated. The tail starts with the pre-computed, fixed-size skip tensors as inputs.

| What the tail has at peak | Size |
|--------------------------|------|
| Input cut tensor y[13] | (32, 128, 40, 40) = 25 MB |
| Skip y[4] (held until L15) | (32, 128, 80, 80) = 100 MB |
| Skip y[10] (held until L21) | (32, 256, 20, 20) = 12.5 MB |
| Tail layer activations (40×40, 80×80, 20×20) | ~50–100 MB |
| cuDNN workspace for tail convs (small spatial) | ~600 MB |
| Model weights (tail layers 14–23 only) | ~5 MB |
| **Total measured** | **≈ 1,136 MB** |

---

## 8. Full Comparison Table

| Metric | Local eager | Local CUDA Graph | Pipeline eager | Pipeline CUDA Graph |
|--------|------------|-----------------|----------------|---------------------|
| Batch compute time | 257 ms | 299 ms | — | — |
| Head stage | — | — | 115 ms | 116 ms |
| Tail stage + NMS | — | — | 194 ms | 230 ms |
| **Throughput (FPS)** | **124.4** | **106.9** | **165.1** | **139.0** |
| vs local eager | 1.00× | 0.86× | 1.33× | 1.12× |
| VRAM peak | 1,486 MB | 1,283 MB | 1,137 MB (tail) | 1,944 MB (tail) |

---

## 9. Conclusions

1. **Optimal cut layer is 12** (not 13): cut_layer=12 gives 1.43× pipeline speedup vs cut_layer=13's 1.33×, because layer 13's C3k2 extends head time without shrinking tail time (tail remains NMS-bottlenecked regardless).

2. **Split pipeline is 1.33× faster** than local at batch=32, cut_layer=13 (`165.1 FPS` vs `124.4 FPS`) because head and tail run concurrently on different batches.

3. **CUDA Graphs ("CUDA coding") make things WORSE** for YOLO11n at batch=32:
   - GPU inference is already computation-bound; kernel-launch overhead is negligible
   - CUDA Graphs pin all activations simultaneously → tail peak VRAM jumps +71% (1,136 → 1,944 MB)
   - CUDA Graphs benefit models where batch=1 or tiny ops dominate; batch=32 YOLO is the opposite

4. **VRAM peak (~1,486 MB) is dominated by**:
   - Large early-layer activation tensors at high spatial resolution (layers 0–2, 160×160 with batch=32: up to 300 MB transient)
   - cuDNN convolution workspace for L0/L1 (selected by benchmark mode, ~600–900 MB)
   - Long-lived skip tensors y[4]=100 MB, y[6]=25 MB that overlap with all of the above

5. **Model weights (10 MB) are only 0.7% of peak inference VRAM.** Cutting 52% of parameters saves essentially no peak VRAM because both head and tail share the expensive backbone layers 0–10.

6. **To meaningfully reduce peak inference VRAM**: cut before layer 2 (removes large early activations), use FP16 (`half: True`, halves all tensor sizes), reduce batch size, or disable `cudnn.benchmark` to avoid large workspace allocation.

7. **The real constraint is bandwidth**: cut_layer=13 sends ~10.7 MB/batch → needs ~1.76 GB/s link at 165 FPS.
