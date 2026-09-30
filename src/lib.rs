//! Change-point detection algorithms.
//!
//! Provides CUSUM, PELT-style, and sliding-window change-point detection.

/// A detected change point.
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub index: usize,
    pub confidence: f64,
}

///CUSUM change-point detection.
pub fn cusum_detect(data: &[f64], threshold: f64, drift: f64) -> Vec<ChangePoint> {
    if data.is_empty() {
        return vec![];
    }
    let mean0 = data[0];
    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;
    let mut changes = vec![];
    for (i, &val) in data.iter().enumerate().skip(1) {
        s_pos = (s_pos + val - mean0 - drift).max(0.0);
        s_neg = (s_neg + mean0 - val - drift).max(0.0);
        if s_pos > threshold || s_neg > threshold {
            changes.push(ChangePoint {
                index: i,
                confidence: s_pos.max(s_neg),
            });
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    changes
}

/// Sliding-window mean-shift detector.
pub fn window_shift_detect(data: &[f64], window_size: usize, threshold: f64) -> Vec<ChangePoint> {
    if data.len() < 2 * window_size {
        return vec![];
    }
    let mut changes = vec![];
    for i in window_size..(data.len() - window_size) {
        let left_mean: f64 = data[i - window_size..i].iter().sum::<f64>() / window_size as f64;
        let right_mean: f64 = data[i..i + window_size].iter().sum::<f64>() / window_size as f64;
        let diff = (right_mean - left_mean).abs();
        if diff > threshold {
            changes.push(ChangePoint { index: i, confidence: diff });
        }
    }
    changes
}

/// Binary segmentation (simplified recursive).
pub fn binary_segmentation(data: &[f64], start: usize, end: usize, threshold: f64) -> Vec<ChangePoint> {
    if end - start < 2 {
        return vec![];
    }
    let mut best_idx = start;
    let mut best_cost = f64::MAX;
    let total_var = {
        let slice = &data[start..end];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
    };
    for k in (start + 1)..end {
        let left = &data[start..k];
        let right = &data[k..end];
        let lm = left.iter().sum::<f64>() / left.len() as f64;
        let rm = right.iter().sum::<f64>() / right.len() as f64;
        let cost = left.iter().map(|x| (x - lm).powi(2)).sum::<f64>()
                 + right.iter().map(|x| (x - rm).powi(2)).sum::<f64>();
        if cost < best_cost {
            best_cost = cost;
            best_idx = k;
        }
    }
    let improvement = total_var - best_cost;
    if improvement < threshold {
        return vec![];
    }
    let mut results = vec![ChangePoint { index: best_idx, confidence: improvement }];
    results.extend(binary_segmentation(data, start, best_idx, threshold));
    results.extend(binary_segmentation(data, best_idx, end, threshold));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_step_change() {
        let mut data = vec![0.0; 50];
        data.extend(vec![5.0; 50]);
        let changes = cusum_detect(&data, 10.0, 0.5);
        assert!(!changes.is_empty());
        assert!(changes[0].index >= 49 && changes[0].index <= 55);
    }

    #[test]
    fn test_window_shift() {
        let mut data = vec![1.0; 20];
        data.extend(vec![10.0; 20]);
        let changes = window_shift_detect(&data, 5, 2.0);
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_binary_segmentation() {
        let mut data = vec![0.0; 30];
        data.extend(vec![10.0; 30]);
        let changes = binary_segmentation(&data, 0, data.len(), 10.0);
        assert!(!changes.is_empty());
    }
}

/// FNV-1a 64 — the digest every substrate in the SuperInstance fleet agrees on.
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// True if this crate's FNV-1a still agrees with the rest of the fleet.
pub fn canary_holds() -> bool {
    fnv1a64("café Δ 日本語".as_bytes()) == 0x024a555471370b18d
}
