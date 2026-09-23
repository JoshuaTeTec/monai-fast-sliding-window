# [Performance] GPU-accelerated fast path for sliding_window_inference using unfold and chunking

Is your feature request related to a problem?
sliding_window_inference is commonly used to run inference on large 3D medical images that cannot be processed as a single volume.
For 3D inputs, the current implementation needs to assemble batches of overlapping windows before passing them to the predictor. This can involve repeated patch materialization, concatenation and memory allocation. The overhead becomes increasingly noticeable as the number of windows grows, for example with larger volumes or higher overlap.

I experimented with an alternative CUDA-specific implementation that keeps the existing behavior as a fallback while reducing some of this window extraction and stitching overhead.

Proposed solution

I propose introducing a guarded CUDA Fast Path for the cases where its assumptions are satisfied.

The prototype uses:

View-based patch extraction using Tensor.unfold(), combined with depth-sliced (Z-axis) chunking.
On-the-fly accumulation using in-place add_ during stitching, avoiding some intermediate allocations.
Dynamic padding to handle input dimensions that do not align with the sliding-window grid.
Support for both constant and gaussian blending.

The Fast Path would only be used for supported cases. Otherwise, the existing sliding_window_inference implementation would be used unchanged.

Scope / fallback behavior

The prototype falls back to the existing implementation when:

the input is not on CUDA (inputs.device.type != "cuda");
a custom blending function is provided instead of "constant" or "gaussian";
the predictor returns a dictionary or a multi-tensor output structure.

This is intended to keep the optimization opt-in by applicability rather than changing the behavior of unsupported cases.

Performance
Environment
	
MONAI	1.6.0
PyTorch	2.11.0+cu128
CUDA	12.8
GPU	Tesla T4
VRAM	14.56 GB
ROI	64 × 64 × 64
Sliding-window batch size	4
1. Sliding-window overhead

To isolate the overhead introduced by window extraction, batching and stitching, I first benchmarked both implementations using an intentionally lightweight predictor (return x.clone()).

| Volume | Overlap | MONAI | Fast Path | Reduction | MONAI peak | Fast peak |
|---|---|---|---|---|---|---|
| 128³ | 0.25 | 3.17 ms | 1.78 ms | 43.9% | 33 MB | 81 MB |
| 128³ | 0.50 | 3.16 ms | 1.63 ms | 48.3% | 33 MB | 42 MB |
| 128³ | 0.75 | 12.64 ms | 6.36 ms | 49.7% | 33 MB | 76 MB |
| 160³ | 0.25 | 3.89 ms | 2.34 ms | 39.8% | 57 MB | 73 MB |
| 160³ | 0.50 | 10.41 ms | 4.86 ms | 53.2% | 57 MB | 80 MB |
| 160³ | 0.75 | 32.74 ms | 16.75 ms | 48.8% | 57 MB | 148 MB |


This benchmark is intended to measure sliding-window orchestration overhead, rather than neural-network inference itself.

The prototype reduces this overhead by approximately 40–53% in these configurations. With these configurations but in gaussian mode, the overhead is reduced by 35%. 

However, the Fast Path currently has a higher peak memory footprint in these tests. The memory/performance trade-off therefore needs to be considered when evaluating the implementation.

CUDA synchronization was performed before and after each measurement, with warm-up iterations before timing. Peak GPU memory was measured using torch.cuda.max_memory_allocated().

2. End-to-end inference

I also benchmarked the implementation with a lightweight 3D segmentation model over a synthetic cohort of 100 volumes.

Volume size: 160³
ROI size: 64³
Overlap: 0.75
Sliding-window batch size: 4
Cohort size: 100 volumes

Outputs were numerically identical (max absolute error = 0)

| Metric | MONAI | Fast Path |
|---|---|---|
| Average time / volume | 188.90 ms | 182.47 ms |
| Average time saved / volume | - | 6.43 ms |
| Time reduction | - | 3.40% |
| Speedup | - | 1.04× |

The Fast Path was faster for all 100 tested volumes. The observed end-to-end improvement is smaller than the isolated sliding-window overhead reduction, as expected since predictor/model computation remains unchanged.

For a hypothetical cohort of 1,000 volumes with the same workload, the measured average would correspond to approximately 6.4 seconds of total time saved.


Hardware considerations

The current benchmark was performed on a Tesla T4, so the results should not be interpreted as representative of all CUDA GPUs.

I would also expect the relative benefit to depend on the ratio between predictor compute and sliding-window overhead. On faster GPUs or with more computationally intensive models, the model itself may dominate runtime and reduce the relative benefit of this optimization.

Conversely, for lightweight models, large volumes, high overlap, or workloads processing many volumes, reducing sliding-window orchestration overhead may have a more noticeable effect.

The current implementation also uses Python-level chunk processing. If this approach proves useful, a future implementation could investigate adaptive chunking based on available GPU memory, or potentially move parts of the operation to a lower-level implementation such as C++/CUDA or Triton.

Current prototype

The prototype currently supports:

CUDA execution;
view-based patch extraction with unfold;
Z-axis chunking;
dynamic padding;
constant blending;
Gaussian blending;
fallback to the existing implementation for unsupported cases.

A complete benchmark notebook and prototype implementation are available here:
[monai_fast_sliding_window_benchmark.ipynb](https://github.com/JoshuaTeTec/monai-fast-sliding-window/blob/main/monai_fast_sliding_window_benchmark.ipynb)

Question for the maintainers

Would the MONAI team be open to this direction as a Pull Request?

In particular, I would be interested in feedback on whether reducing the sliding-window orchestration overhead through a guarded CUDA Fast Path is compatible with the project's design and performance goals.

If this approach seems useful, I would be happy to prepare a PR with the implementation, tests, correctness checks, and benchmarks.
