# Change-Point Detection — Statistical Regime Shift Detection

**Change-point detection** identifies indices in a time series where the statistical distribution changes — shifts in mean, variance, or distributional shape. This crate implements three classical algorithms: CUSUM, sliding-window mean-shift, and binary segmentation.

## Why It Matters

Change-point detection is fundamental to monitoring, quality control, and signal processing. In infrastructure monitoring, a change-point marks the moment latency spikes or error rates jump — the exact boundary between "healthy" and "degraded." In A/B testing, it separates regimes. In finance, it flags regime shifts in volatility. In manufacturing, it catches sensor drift. Engineers building observability platforms, anomaly detectors, and automated incident responders all need change-point detection to answer: *when did things go wrong?*

## How It Works

### CUSUM (Cumulative Sum)

CUSUM tracks the cumulative deviation of observations from a baseline mean, with a drift parameter `k` that absorbs random noise:

```
S⁺(i) = max(0, S⁺(i-1) + xᵢ - μ₀ - k)
S⁻(i) = max(0, S⁻(i-1) + μ₀ - xᵢ - k)
```

If either `S⁺` or `S⁻` exceeds a threshold `h`, a change-point is signaled and the statistic resets to 0. The drift `k` is typically set to half the expected shift magnitude. CUSUM is optimal for detecting sustained shifts — it has the smallest average detection delay among all tests with a fixed false-alarm rate (Lorden, 1971). Complexity: `O(n)` for a series of length `n`.

### Sliding-Window Mean-Shift Detector

Computes the mean of a trailing window `[i-w, i)` and a leading window `[i, i+w)`, then flags positions where `|mean_right - mean_left| > threshold`. This is a nonparametric test — no distributional assumptions. Complexity: `O(n · w)` naively (the implementation recomputes means per position). For large windows, a running-sum optimization reduces this to `O(n)`.

### Binary Segmentation

A recursive divide-and-conquer approach. At each step, find the split point `k*` that minimizes within-segment variance:

```
k* = argmin_k  [ SS(X[start:k]) + SS(X[k:end]) ]
```

where `SS(·)` is the sum of squared deviations from the segment mean. If the improvement `SS(X[start:end]) - SS(X[start:k*]) - SS(X[k*:end])` exceeds a threshold, recurse on both halves. Binary segmentation is `O(n log n)` on average and `O(n²)` in the worst case (when every point is a candidate split). It approximates the exact dynamic programming approach (PELT, `O(n)`) at a fraction of the implementation complexity.

## Quick Start

```rust
use change_point::{cusum_detect, window_shift_detect, binary_segmentation};

// Build a signal with a step change at index 50
let mut data = vec![0.0; 50];
data.extend(vec![5.0; 50]);

// CUSUM: detects the shift within a few samples
let changes = cusum_detect(&data, threshold: 10.0, drift: 0.5);
assert!(changes[0].index >= 49 && changes[0].index <= 55);

// Sliding window: compares left/right means
let changes = window_shift_detect(&data, window_size: 5, threshold: 2.0);
assert!(!changes.is_empty());

// Binary segmentation: recursive multi-change-point detection
let changes = binary_segmentation(&data, 0, data.len(), 10.0);
assert!(!changes.is_empty());
```

## API

| Function | Signature | Complexity |
|---|---|---|
| `cusum_detect(data, threshold, drift)` | `→ Vec<ChangePoint>` | `O(n)` |
| `window_shift_detect(data, window_size, threshold)` | `→ Vec<ChangePoint>` | `O(n · w)` |
| `binary_segmentation(data, start, end, threshold)` | `→ Vec<ChangePoint>` | `O(n log n)` avg |

| Type | Description |
|---|---|
| `ChangePoint` | `{ index: usize, confidence: f64 }` — position and detection statistic. |

## Architecture Notes

Change-point detection is part of the η (evaluation/analysis) pipeline in γ + η = C. It processes metric streams from the SuperInstance fleet to detect incidents, capacity shifts, and performance regressions without manual thresholding. See [SuperInstance Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika 41(1/2), 100–115. — Original CUSUM paper.
2. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal Detection of Changepoints with a Linear Computational Cost*. JASA 107(500), 1590–1598. — PELT algorithm.
3. Basseville, M. & Nikiforov, I. V. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice-Hall.

## License

MIT
