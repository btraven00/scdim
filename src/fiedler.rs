//! Algebraic connectivity and the eigengap, from the graph Laplacian.
//!
//! Build a k-NN graph on the cells, take the normalised Laplacian
//! `L = I - D^-1/2 W D^-1/2`, and read its bottom eigenvalues:
//!
//! - `lambda_1`, the **Fiedler value**. Zero exactly when the graph is
//!   disconnected; near zero when it is two patches joined by a few weak
//!   edges. This is the binary single-vs-multiple call.
//! - The **eigengap**: if there are k patches then `lambda_0 .. lambda_k-1`
//!   sit near zero and `lambda_k` jumps. The largest jump locates k.
//!
//! The eigenvalues of the normalised Laplacian live in `[0, 2]` whatever the
//! degrees are, so `lambda_1` is comparable across datasets in a way the
//! unnormalised `D - W` is not.
//!
//! No sparse eigensolver: at the point counts these diagnostics run on
//! (<= a few thousand cells) the dense Laplacian is the same size as the
//! covariance EVD already being done, and a dense symmetric EVD hands back the
//! whole bottom spectrum -- so the eigengap comes free next to the Fiedler
//! value instead of needing a second ARPACK call for each extra eigenvalue.
//!
//! Edge weights are self-tuning (Zelnik-Manor & Perona 2004): the local scale
//! `sigma_i` is the distance to the k-th neighbour, and
//! `W_ij = exp(-d_ij^2 / (sigma_i sigma_j))`. A single global bandwidth is the
//! wrong instrument for single-cell data, where a dense cell type and a sparse
//! one sit in the same embedding.

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Mat, Side};

use crate::geom::Cloud;
use crate::rank::Estimate;

/// Bottom eigenvalues of the normalised Laplacian of the k-NN graph,
/// ascending. Returns the whole spectrum; callers look at the first few.
///
/// `nbr` is the shared k-NN graph, `(d^2, index)` ascending, so the local
/// scale sigma_i is just its last entry.
pub fn laplacian_spectrum(c: &Cloud, nbr: &[Vec<(f64, usize)>]) -> Vec<f64> {
    let m = c.len();
    if nbr.len() != m || m < 3 {
        return Vec::new();
    }
    let sigma: Vec<f64> = nbr
        .iter()
        .map(|n| n.last().expect("k >= 1").0.sqrt().max(1e-12))
        .collect();

    // Symmetric self-tuning affinity: W = max(W, W^T), so a one-directional
    // neighbour still connects. Union rather than intersection -- the
    // intersection disconnects sparse regions and manufactures patches.
    let mut w = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for &(dd, j) in &nbr[i] {
            let aff = (-dd / (sigma[i] * sigma[j])).exp();
            if aff > w.read(i, j) {
                w.write(i, j, aff);
                w.write(j, i, aff);
            }
        }
    }

    let deg: Vec<f64> = (0..m)
        .map(|i| (0..m).map(|j| w.read(i, j)).sum::<f64>())
        .collect();
    // L = I - D^-1/2 W D^-1/2. An isolated node keeps its identity row, which
    // gives it eigenvalue 1 rather than a NaN.
    let l = Mat::from_fn(m, m, |i, j| {
        let ident = if i == j { 1.0 } else { 0.0 };
        if deg[i] <= 0.0 || deg[j] <= 0.0 {
            return ident;
        }
        ident - w.read(i, j) / (deg[i] * deg[j]).sqrt()
    });

    let evd = SelfAdjointEigendecomposition::new(l.as_ref(), Side::Lower);
    (0..m).map(|i| evd.s().column_vector().read(i)).collect()
}

/// The Fiedler value and the eigengap, as one verdict.
///
/// # Why lambda_1 alone cannot make the call
///
/// "lambda_1 near zero means partitioned" has a false positive, and it fires on
/// exactly the data this crate cares about. A path graph on N nodes has
/// algebraic connectivity ~(pi/N)^2 while being perfectly connected: measured
/// here, 800 points along a smooth 1-D curve give lambda_1 = 1.4e-4, which any
/// absolute threshold reads as "partitioned". A trajectory is a long thin
/// manifold, so it will always look weakly connected in absolute terms.
///
/// So the verdict rests on two things that do not have that failure mode:
///
/// - **Exact zeros.** The multiplicity of eigenvalue 0 is the number of
///   connected components, exactly, with no threshold to tune. This is the
///   unambiguous partition test.
/// - **The relative eigengap** `lambda_k+1 / lambda_k`. For k separated
///   clusters the ratio at k is enormous (the denominator is ~0); for a path
///   graph `lambda_k ~ (k pi/N)^2`, so consecutive ratios are
///   `(k+1)^2/k^2` = 4, 2.25, 1.78, ... -- bounded and shrinking. A ratio
///   threshold separates the two cases where an absolute one cannot.
///
/// `lambda_1` is still reported: as algebraic connectivity it is a real measure
/// of how thin the bottleneck is. It is just not, on its own, a classifier.
pub fn fiedler(eigs: &[f64], max_patches: usize, min_ratio: f64) -> Estimate {
    if eigs.len() < 4 {
        return Estimate {
            name: "fiedler",
            rank: 0,
            detail: "too few points for a k-NN graph".to_string(),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    const ZERO: f64 = 1e-8;
    let components = eigs.iter().take_while(|&&e| e < ZERO).count().max(1);
    let lambda1 = eigs[1];

    // Largest relative jump in the low spectrum, over the connected part.
    let hi = max_patches.min(eigs.len() - 1);
    let (mut best_k, mut best_ratio) = (1usize, 1.0f64);
    for k in components.max(1)..hi {
        if eigs[k] > ZERO {
            let r = eigs[k + 1] / eigs[k];
            if r > best_ratio {
                best_ratio = r;
                best_k = k + 1;
            }
        }
    }

    if components > 1 {
        return Estimate {
            name: "fiedler",
            rank: components,
            detail: format!(
                "PARTITIONED: {components} exact connected components (lambda_1 = \
                 {lambda1:.2e}); next relative eigengap {best_ratio:.2}x at k = {best_k}"
            ),
            stat: Some(best_ratio),
            pvalues: Vec::new(),
        };
    }
    if best_ratio >= min_ratio {
        return Estimate {
            name: "fiedler",
            rank: best_k,
            detail: format!(
                "{best_k} weakly-joined patches: connected (lambda_1 = {lambda1:.2e}) but \
                 relative eigengap {best_ratio:.2}x at k = {best_k} (>= {min_ratio})"
            ),
            stat: Some(best_ratio),
            pvalues: Vec::new(),
        };
    }
    Estimate {
        name: "fiedler",
        rank: 1,
        detail: format!(
            "connected: lambda_1 = {lambda1:.2e}, largest relative eigengap only \
             {best_ratio:.2}x (< {min_ratio}) -- one patch. Note a small lambda_1 alone \
             means thin, not split: a path graph gives ~(pi/N)^2"
        ),
        stat: Some(best_ratio),
        pvalues: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// One cloud, one k-NN graph, one spectrum -- the wiring `main` uses.
    fn spectrum(x: &Mat<f64>, max_points: usize, k: usize) -> Vec<f64> {
        let c = Cloud::new(x, max_points);
        laplacian_spectrum(&c, &c.knn(k))
    }

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Three well-separated blobs: the k-NN graph really is disconnected, so
    /// lambda_1 is an exact zero and the eigengap sits at 3.
    #[test]
    fn separated_blobs_are_partitioned() {
        let mut r = lcg(11);
        let x = Mat::from_fn(600, 8, |i, j| {
            let blob = i % 3;
            (if j == blob { 100.0 } else { 0.0 }) + r()
        });
        let eigs = spectrum(&x, 600, 15);
        let est = fiedler(&eigs, 20, 5.0);
        assert!(eigs[1] < 1e-8, "lambda_1 = {:.3e}", eigs[1]);
        assert_eq!(est.rank, 3, "{}", est.detail);
        assert!(est.detail.starts_with("PARTITIONED"), "{}", est.detail);
    }

    /// The case that breaks a raw lambda_1 threshold: a connected 1-D curve
    /// has lambda_1 ~ (pi/N)^2, tiny but not zero. The verdict must still be
    /// one patch.
    #[test]
    fn curve_is_connected() {
        let x = Mat::from_fn(800, 6, |i, j| {
            let t = i as f64 / 800.0 * 8.0;
            match j {
                0 => t.cos(),
                1 => t.sin(),
                2 => t / 8.0,
                _ => 0.05 * (t * j as f64).cos(),
            }
        });
        let eigs = spectrum(&x, 800, 15);
        let est = fiedler(&eigs, 20, 5.0);
        assert!(eigs[1] > 0.0 && eigs[1] < 1e-3, "lambda_1 = {:.3e}", eigs[1]);
        assert_eq!(est.rank, 1, "{}", est.detail);
        assert!(est.detail.starts_with("connected"), "{}", est.detail);
    }

    /// The normalised Laplacian's spectrum must start at zero and stay in
    /// [0, 2] -- the property that makes lambda_1 comparable across datasets.
    #[test]
    fn spectrum_is_normalised() {
        let mut r = lcg(5);
        let x = Mat::from_fn(400, 10, |_, _| r());
        let eigs = spectrum(&x, 400, 10);
        assert_eq!(fiedler(&eigs, 20, 5.0).rank, 1);
        assert!(eigs[0].abs() < 1e-9, "lambda_0 = {:.3e}", eigs[0]);
        assert!(eigs.windows(2).all(|w| w[1] >= w[0] - 1e-12));
        assert!(eigs.iter().all(|&e| (-1e-9..=2.0 + 1e-9).contains(&e)));
    }
}
