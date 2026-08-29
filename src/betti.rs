//! Continuous manifold or discrete patches? The 0-th Betti number, read off
//! the minimum spanning tree.
//!
//! The 0-dimensional persistent homology of a Vietoris-Rips filtration is
//! single-linkage clustering, and single-linkage is the Euclidean minimum
//! spanning tree. The H_0 barcode's death times *are* the MST edge weights:
//! two components merge at radius r exactly when an MST edge of weight r is
//! added. So
//!
//!   beta_0(r) = N - #{ MST edges <= r }
//!
//! is the entire barcode, and Prim's algorithm computes it in O(N^2) over the
//! same Gram matrix the other neighbour-based heuristics already build. No
//! simplicial complex is ever constructed, and no persistent-homology library
//! is needed -- for H_0 they would compute this and nothing more.
//!
//! ## What it diagnoses
//!
//! Sort the MST weights. On a connected manifold, sampled densely, they are
//! unimodal and the largest is unremarkable: every point has a near neighbour
//! and beta_0 decays smoothly. On well-separated patches, the few edges that
//! bridge the patches are far longer than the rest, and the sorted weights show
//! a step. The size of that step is the diagnostic:
//!
//!   gap = max over the upper tail of w[i+1] / w[i]
//!
//! and the number of edges above the gap, plus one, is the patch count. This is
//! the same reasoning as the Laplacian eigengap, but it needs no k-NN graph, no
//! choice of k, and no sparse eigensolver -- and unlike the Fiedler value it
//! does not assume the answer is two.
//!
//! Ambient RNA and doublets are exactly what puts weak bridges between real
//! patches, which is why the interesting quantity is the *ratio* at the step
//! rather than a count of literal zero eigenvalues: the bridges make the
//! components formally connected while leaving the step intact.

use crate::geom::Cloud;
use crate::rank::Estimate;

/// Edge weights of the Euclidean MST, ascending. Equivalently the H_0 barcode:
/// the i-th weight is the radius at which the (N-i)-th component dies.
pub fn mst_weights(c: &Cloud) -> Vec<f64> {
    let m = c.len();
    if m < 3 {
        return Vec::new();
    }
    let dist = |a: usize, b: usize| c.dist(a, b);

    // Prim's, dense: no priority queue, because the graph is complete and the
    // O(m^2) scan is the same cost as reading the distances once.
    let mut in_tree = vec![false; m];
    let mut best = vec![f64::INFINITY; m];
    let mut out = Vec::with_capacity(m - 1);
    best[0] = 0.0;
    for k in 0..m {
        let u = (0..m)
            .filter(|&i| !in_tree[i])
            .min_by(|&a, &b| best[a].total_cmp(&best[b]))
            .expect("m nodes, m iterations");
        in_tree[u] = true;
        if k > 0 {
            out.push(best[u]); // the edge that attached u: one H_0 death
        }
        for v in 0..m {
            if !in_tree[v] {
                best[v] = best[v].min(dist(u, v));
            }
        }
    }
    out.retain(|w| w.is_finite());
    out.sort_by(f64::total_cmp);
    out
}

/// Number of patches, from the largest relative step in the MST weights.
///
/// `max_patches` bounds how far down the sorted weights the step is looked
/// for: a step near the bottom would mean thousands of singleton components,
/// which is a statement about sampling density, not about structure.
///
/// A gap ratio below `min_gap` is reported as one connected manifold. The
/// default of 2.0 says the bridging edge has to be twice the next-longest
/// before "these are separate patches" beats "this is one lumpy manifold".
pub fn patch_count(weights: &[f64], max_patches: usize, min_gap: f64) -> Estimate {
    let l = weights.len();
    if l < 10 {
        return Estimate {
            name: "betti0",
            rank: 0,
            detail: "too few points for an MST".to_string(),
            pvalues: Vec::new(),
        };
    }
    let lo = l.saturating_sub(max_patches.max(2));
    let (mut best_i, mut best_ratio) = (l - 2, 1.0f64);
    for i in lo..l - 1 {
        if weights[i] > 0.0 {
            let r = weights[i + 1] / weights[i];
            if r > best_ratio {
                best_ratio = r;
                best_i = i;
            }
        }
    }
    let patches = l - best_i;
    let median = weights[l / 2];
    let spread = weights[l - 1] / median;

    if best_ratio < min_gap {
        return Estimate {
            name: "betti0",
            rank: 1,
            detail: format!(
                "continuous: largest MST step is {best_ratio:.2}x (< {min_gap}), longest edge \
                 {spread:.1}x the median -- one connected manifold, no separated patches"
            ),
            pvalues: Vec::new(),
        };
    }
    Estimate {
        name: "betti0",
        rank: patches,
        detail: format!(
            "{patches} patches: MST step {best_ratio:.2}x at edge {}/{l} (r = {:.3}), longest \
             edge {spread:.1}x the median",
            best_i + 1,
            weights[best_i + 1]
        ),
        pvalues: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Three tight, well-separated blobs are three patches, and the step in the
    /// MST weights is what says so.
    #[test]
    fn finds_separated_blobs() {
        let mut r = lcg(11);
        let x = Mat::from_fn(600, 8, |i, j| {
            let blob = i % 3;
            let centre = if j == blob { 100.0 } else { 0.0 };
            centre + r()
        });
        let w = mst_weights(&Cloud::new(&x, 600));
        let est = patch_count(&w, 20, 2.0);
        assert_eq!(est.rank, 3, "{}", est.detail);
    }

    /// One continuous curve is one manifold, however it is embedded: no step,
    /// so the verdict must be "continuous" and not a patch count.
    #[test]
    fn calls_a_curve_continuous() {
        let x = Mat::from_fn(800, 6, |i, j| {
            let t = i as f64 / 800.0 * 8.0;
            match j {
                0 => t.cos(),
                1 => t.sin(),
                2 => t / 8.0,
                _ => 0.05 * (t * j as f64).cos(),
            }
        });
        let est = patch_count(&mst_weights(&Cloud::new(&x, 800)), 20, 2.0);
        assert_eq!(est.rank, 1, "{}", est.detail);
        assert!(est.detail.starts_with("continuous"), "{}", est.detail);
    }

    /// A uniform cloud has no structure to find either -- guards against the
    /// gap search manufacturing patches out of the tail of a smooth
    /// distribution.
    #[test]
    fn calls_a_gaussian_blob_continuous() {
        let mut r = lcg(5);
        let x = Mat::from_fn(800, 10, |_, _| r() + r() + r() - 1.5);
        let est = patch_count(&mst_weights(&Cloud::new(&x, 800)), 20, 2.0);
        assert_eq!(est.rank, 1, "{}", est.detail);
    }

    /// The barcode has to be a barcode: N-1 finite, sorted, non-negative
    /// weights.
    #[test]
    fn mst_is_well_formed() {
        let mut r = lcg(3);
        let x = Mat::from_fn(200, 5, |_, _| r());
        let w = mst_weights(&Cloud::new(&x, 200));
        assert_eq!(w.len(), 199);
        assert!(w.windows(2).all(|p| p[1] >= p[0] && p[0] >= 0.0));
    }
}
