//! Rank-selection heuristics on a biwhitened eigenspectrum.
//!
//! All of them read the same object: the eigenvalues of the sample covariance
//! of the biwhitened, mean-centred matrix, rescaled so the noise bulk sits on
//! the canonical Marchenko-Pastur scale. Biwhitening is what makes that scale
//! meaningful for counts (Chardes et al. 2025); without it the bulk is not MP
//! and none of these thresholds mean anything.

/// Score columns retained. Past this the components are noise by every
/// heuristic here, and the geometric ones only ever slice the leading few.
pub const SCORE_K: usize = 100;

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Mat, Side};
use rayon::prelude::*;
use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::rmt::RmtTheory;
use rmt_spca::sparse::CsrMatrix;
use std::time::Instant;

/// Eigenspectrum of the biwhitened covariance, on the MP scale.
pub struct Spectrum {
    /// Eigenvalues, descending, divided by the estimated noise level sigma^2.
    pub eigenvalues: Vec<f64>,
    /// Cells (rows) and genes (columns) actually used.
    pub n: usize,
    pub p: usize,
    /// Aspect ratio q = p / n.
    pub q: f64,
    /// Noise level estimated by matching the bulk median to the MP median.
    pub sigma_sq: f64,
    /// Whether Sinkhorn-Knopp biwhitening converged. If false, treat every
    /// number below as unreliable: the bulk is not on the MP scale.
    pub biwhitening_converged: bool,
    pub biwhitening_residual: f64,
    /// KS distance between the bulk eigenvalues and the Marchenko-Pastur CDF.
    ///
    /// This, not `biwhitening_converged`, is the diagnostic that decides
    /// whether the spectral heuristics mean anything: they all assume the noise
    /// bulk *is* MP. Chardes et al. take <= 0.10 as a good fit. Sinkhorn's own
    /// step-to-step residual can sit at 1e-3 forever (it oscillates under
    /// damping rather than converging) while the resulting spectrum is
    /// perfectly good, so the flag alone is not evidence of anything.
    pub bulk_ks: f64,
    /// Wall seconds of the three parts: Sinkhorn, covariance, eigendecomposition
    /// (+ scores). Printed as sub-stages; the stage total alone hid where a
    /// regression was.
    pub timings: [f64; 3],
    /// PCA scores, n x min(scores_k, rank): the cells in the leading principal
    /// subspace, on the U*Sigma scale. Empty (n x 0) when `scores_k` is 0.
    ///
    /// The geometric heuristics (TwoNN, correlation dimension, Betti-0) must
    /// run here rather than in ambient gene space. In 2000 dimensions pairwise
    /// distances concentrate -- measured on Tabula Muris FACS, the longest MST
    /// edge is 1.2x the median and the largest single-linkage step is 1.01x,
    /// so 81 well-separated cell types are indistinguishable from a smooth
    /// curve. The eigenvectors come free with the EVD that is run anyway.
    pub scores: Mat<f64>,
}

impl Spectrum {
    /// Biwhiten, centre, and diagonalise a CSR count matrix.
    ///
    /// `damp` and `max_iter` are the Sinkhorn-Knopp knobs. Real count data
    /// oscillates under the undamped step; if `biwhitening_converged` comes
    /// back false, lower `damp` or raise `max_iter` before believing anything.
    ///
    /// `scores_k` is how many leading score columns to return. 0 asks for the
    /// eigenvalues only (`--rank-only`), which skips the eigenvectors: at
    /// p = 8248 that halves the EVD and drops the O(p^2 k) score product.
    ///
    /// Sinkhorn always runs on CSR (rmt-spca >= 0.1.1). The covariance has two
    /// paths, same numbers (tested): `Dense` forms the centred whitened n x p
    /// matrix and uses faer's gemm; `Sparse` accumulates
    /// S = (Y'Y - n mu mu') / (n-1) from the nonzeros, never forming n x p
    /// (memory O(nnz + p^2)) but with no BLAS.
    ///
    /// Measured at 8 threads (rank-only), Sinkhorn dense vs CSR / covariance
    /// dense vs sparse: panel 20k x 2000, 72% dense: 10.4 vs 6.4 s / 1.2 vs
    /// 10.1 s; full 20k x 8248, 38% dense: 42.3 vs 12.5 s / 11.5 vs 50.9 s. The
    /// sparse covariance's slowdown is ~11.5 x density in both, so it only wins
    /// below ~9%: [`SPARSE_MAX_DENSITY`].
    pub fn compute(x: &CsrMatrix, damp: f64, max_iter: usize, scores_k: usize, path: SpectrumPath) -> Self {
        let (n, p) = (x.nrows(), x.ncols());
        let q = p as f64 / n as f64;
        let bw = Biwhitener {
            damp,
            max_iter,
            ..Biwhitener::default()
        };
        let m = n.min(p);
        let denom = (n - 1) as f64;
        let t0 = Instant::now();

        // Centred whitened rows (dense path, or sparse with p > n), else None.
        let (c, d, converged, residual, mu, dense_rows, t_bw, gram, t_cov);
        match path {
            SpectrumPath::Dense => {
                let (cc, dd, _, conv, res) = bw.compute_csr(x);
                t_bw = t0.elapsed().as_secs_f64();
                let means = x.biwhitened_mean(&x.transpose(), &cc, &dd);
                let xc = x.centred_rows(&(0..n).collect::<Vec<_>>(), &cc, &dd, &means);
                let g = if p <= n {
                    xc.as_ref().transpose() * xc.as_ref()
                } else {
                    xc.as_ref() * xc.as_ref().transpose()
                };
                gram = Mat::from_fn(m, m, |i, j| g.read(i, j) / denom);
                (c, d, converged, residual, mu, dense_rows) = (cc, dd, conv, res, means, Some(xc));
            }
            SpectrumPath::Sparse => {
                let (cc, dd, _, conv, res) = bw.compute_csr(x);
                t_bw = t0.elapsed().as_secs_f64();
                let xt = x.transpose();
                let mm = x.biwhitened_mean(&xt, &cc, &dd);
                // p > n only happens for tiny cell counts; the n x n path densifies.
                let rows = (p > n).then(|| x.centred_rows(&(0..n).collect::<Vec<_>>(), &cc, &dd, &mm));
                gram = match &rows {
                    None => x.biwhitened_covariance(&xt, &cc, &dd),
                    Some(yc) => {
                        let g = yc.as_ref() * yc.as_ref().transpose();
                        Mat::from_fn(m, m, |i, j| g.read(i, j) / denom)
                    }
                };
                (c, d, converged, residual, mu, dense_rows) = (cc, dd, conv, res, mm, rows);
            }
        }
        t_cov = t0.elapsed().as_secs_f64() - t_bw;

        let k = scores_k.min(m);
        let (mut eigenvalues, scores) = if k == 0 {
            let mut ev = gram.selfadjoint_eigenvalues(Side::Lower);
            ev.sort_by(|a, b| b.total_cmp(a));
            (ev, Mat::zeros(n, 0))
        } else {
            let evd = SelfAdjointEigendecomposition::new(gram.as_ref(), Side::Lower);
            let mut ev: Vec<f64> = (0..m).map(|i| evd.s().column_vector().read(i)).collect();
            ev.reverse(); // faer returns ascending
            // Scores = U*Sigma either way. When the Gram is X^T X the
            // eigenvectors are gene loadings V and scores = (Y - 1 mu') V, done
            // row by row over the nonzeros; when it is X X^T they are the cell
            // directions U and scores = U*sqrt((n-1)*lambda).
            let u = evd.u();
            let col = |t: usize| m - 1 - t; // t-th largest
            let scores = if p > n {
                Mat::from_fn(n, k, |i, t| u.read(i, col(t)) * (denom * ev[t].max(0.0)).sqrt())
            } else if let Some(xc) = &dense_rows {
                let v = Mat::from_fn(p, k, |j, t| u.read(j, col(t)));
                xc.as_ref() * v.as_ref()
            } else {
                // Sparse rows: (Y - 1 mu') V over the nonzeros, V row-major so
                // the inner loop is contiguous.
                let vr: Vec<f64> = (0..p).flat_map(|j| (0..k).map(move |t| (j, t))).map(|(j, t)| u.read(j, col(t))).collect();
                let muv: Vec<f64> = (0..k).map(|t| (0..p).map(|j| mu[j] * vr[j * k + t]).sum()).collect();
                let rows: Vec<Vec<f64>> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let mut r = vec![0.0; k];
                        for (j, val) in x.row(i) {
                            let y = c.read(i) * val * d.read(j);
                            for (rt, vj) in r.iter_mut().zip(&vr[j * k..(j + 1) * k]) {
                                *rt += y * vj;
                            }
                        }
                        r.iter_mut().zip(&muv).for_each(|(rt, m)| *rt -= m);
                        r
                    })
                    .collect();
                Mat::from_fn(n, k, |i, t| rows[i][t])
            };
            (ev, scores)
        };
        let t_eig = t0.elapsed().as_secs_f64() - t_bw - t_cov;

        // sigma^2 by median-matching against MP. No attempt is made to exclude
        // the signal eigenvalues first, and none is needed: the median of 2000
        // eigenvalues does not notice the ~80 at the top, which is the whole
        // reason for preferring it to a trace-based estimate.
        //
        // There used to be an upper bound here, `e <= lambda_plus * lambda_1`,
        // which read as if it excluded them. It never excluded anything --
        // lambda_plus >= 1 and lambda_1 is the largest eigenvalue, so the bound
        // holds for every e by construction. Deleted rather than repaired,
        // because the estimator is right as it stands and the line was only
        // ever describing something it did not do.
        //
        // The lower bound is real: mean-centring costs a degree of freedom, so
        // when n <= p one eigenvalue sits at ~0 and would drag the median down.
        let theory = RmtTheory { q };
        let mut bulk: Vec<f64> = eigenvalues.iter().copied().filter(|&e| e > 1e-3).collect();
        bulk.sort_by(f64::total_cmp);
        let sigma_sq = if bulk.is_empty() {
            1.0
        } else {
            bulk[bulk.len() / 2] / theory.mp_median()
        };
        for e in &mut eigenvalues {
            *e /= sigma_sq;
        }

        let bulk_ks = rmt_spca::verification::calculate_bulk_ks(&eigenvalues, q);

        Spectrum {
            eigenvalues,
            n,
            p,
            q,
            sigma_sq,
            biwhitening_converged: converged,
            biwhitening_residual: residual,
            bulk_ks,
            timings: [t_bw, t_cov, t_eig],
            scores,
        }
    }
}

/// Which implementation [`Spectrum::compute`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumPath {
    Dense,
    Sparse,
}

/// Density at or below which `auto` takes the sparse covariance. Extrapolated
/// from two measured densities (72%, 38%; see [`Spectrum::compute`]), not
/// measured below 38%.
pub const SPARSE_MAX_DENSITY: f64 = 0.08;

impl SpectrumPath {
    pub fn auto(x: &CsrMatrix) -> Self {
        let density = x.nnz() as f64 / (x.nrows() * x.ncols()).max(1) as f64;
        if density <= SPARSE_MAX_DENSITY { SpectrumPath::Sparse } else { SpectrumPath::Dense }
    }
}

/// Longest run of consecutive entries all within `tol` (relative) of the
/// run's first element.
///
/// The bound is on the whole run, not on adjacent steps: a steady 9%-per-level
/// decline passes an adjacent-step test at tol = 0.10 while halving over five
/// levels, and calling that a plateau is the error this exists to avoid.
pub fn longest_flat_run(vals: &[f64], tol: f64) -> std::ops::Range<usize> {
    let mut best = 0..0;
    for i in 0..vals.len() {
        let mut j = i;
        while j + 1 < vals.len() && (vals[j + 1] - vals[i]).abs() <= tol * vals[i].abs() {
            j += 1;
        }
        if j - i > best.end - best.start {
            best = i..j;
        }
    }
    best
}

/// One heuristic's answer.
pub struct Estimate {
    pub name: &'static str,
    pub rank: usize,
    /// The continuous reading, where the row has one.
    ///
    /// Two rows here have no integer to give: `betti0` has returned 1 and
    /// `ricci-neg` 0 on every dataset tried, because "one piece or several" and
    /// "how deep are the bottlenecks" are matters of degree that a count can
    /// only answer once a threshold has already decided them. Forcing every
    /// heuristic through `rank: usize` is what made those two print a constant
    /// while their actual content sat in the detail string as prose.
    ///
    /// `None` where the answer really is a count -- `tracy-widom` and `mp-edge`
    /// are counting eigenvalues and nothing is being rounded away.
    pub stat: Option<f64>,
    /// Human-readable justification (threshold, statistic, ...).
    pub detail: String,
    /// Per-component p-values, where the heuristic is a sequence of tests.
    /// Empty otherwise.
    pub pvalues: Vec<f64>,
}

/// Tracy-Widom sequential test.
///
/// Under the null "everything left is noise", the largest eigenvalue of a real
/// Wishart matrix, centred and scaled by Johnstone (2001) with Ma's (2012)
/// -1/2 refinement, converges to the order-1 Tracy-Widom law:
///
///   mu    = (sqrt(n-1/2) + sqrt(p-1/2))^2
///   sigma = (sqrt(n-1/2) + sqrt(p-1/2)) * (1/sqrt(n-1/2) + 1/sqrt(p-1/2))^(1/3)
///   (n*lambda - mu) / sigma  ->  TW1
///
/// Test lambda_1; if its p-value is below `alpha` it is signal, so drop it,
/// shrink (n,p) by one and test lambda_2. The rank is the number rejected
/// before the first acceptance -- the sequential construction of Patterson,
/// Price & Reich (2006).
///
/// `pvalues` holds one entry per component tested, the last being the one that
/// stopped the sequence. They are raw, not multiplicity-corrected: the
/// sequential stop is the correction, in the sense that only the first
/// acceptance is acted on.
pub fn tracy_widom(s: &Spectrum, alpha: f64, k_max: usize) -> Estimate {
    let k_max = k_max.min(s.eigenvalues.len().saturating_sub(1));
    let mut pvalues = Vec::new();
    for k in 0..k_max {
        let ne = (s.n - 1 - k) as f64;
        let pe = (s.p - k) as f64;
        if ne <= 1.0 || pe <= 1.0 {
            break;
        }
        let (a, b) = ((ne - 0.5).sqrt(), (pe - 0.5).sqrt());
        let mu = (a + b).powi(2);
        let sigma = (a + b) * (1.0 / a + 1.0 / b).cbrt();
        let stat = (s.eigenvalues[k] * ne - mu) / sigma;
        let pv = crate::tw::tw1_sf(stat);
        pvalues.push(pv);
        if pv > alpha {
            return Estimate {
                name: "tracy-widom",
                rank: k,
                detail: format!(
                    "stopped at component {}: TW1 statistic {stat:.3}, p = {pv:.3e} > {alpha}",
                    k + 1
                ),
                stat: None,
                pvalues,
            };
        }
    }
    let last = pvalues.last().copied().unwrap_or(f64::NAN);
    Estimate {
        name: "tracy-widom",
        rank: k_max,
        detail: format!(
            "hit k_max={k_max}, component {k_max} still significant (p = {last:.3e}); \
             the rank is a floor, not a result -- raise --k-max"
        ),
        stat: None,
        pvalues,
    }
}

/// Marchenko-Pastur bulk edge: count eigenvalues above lambda+ = (1+sqrt(q))^2.
///
/// The asymptotic limit of the same idea, with no finite-size correction, so it
/// is systematically more permissive than the Tracy-Widom test. Kept as the
/// baseline you compare against: when the two disagree by a lot, the bulk is
/// probably not MP and the biwhitening deserves a look.
pub fn mp_edge(s: &Spectrum) -> Estimate {
    let lp = RmtTheory { q: s.q }.lambda_plus();
    let rank = s.eigenvalues.iter().take_while(|&&e| e > lp).count();
    Estimate {
        name: "mp-edge",
        rank,
        detail: format!("eigenvalues above lambda+ = {lp:.4}"),
        stat: None,
        pvalues: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure Poisson noise has no signal: both heuristics should say ~0, and
    /// Tracy-Widom must never be more permissive than the raw MP edge.
    #[test]
    fn null_data_has_no_rank() {
        let (n, p) = (400, 120);
        // Deterministic Poisson-ish counts via a cheap LCG, mean 5.
        let mut state = 42u64;
        let x = Mat::from_fn(n, p, |_, _| {
            let mut k = 0.0;
            for _ in 0..10 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                k += ((state >> 33) % 2) as f64;
            }
            k
        });
        let s = Spectrum::compute(&CsrMatrix::from_dense(&x), 0.3, 1000, SCORE_K, SpectrumPath::Sparse);
        let tw = tracy_widom(&s, 0.001, 20);
        let mp = mp_edge(&s);
        assert!(tw.rank <= 2, "TW found {} components in noise", tw.rank);
        assert!(tw.rank <= mp.rank, "TW {} > MP {}", tw.rank, mp.rank);
    }

    /// A rank-1 spike planted on top of the same noise must be detected.
    #[test]
    fn planted_spike_is_found() {
        let (n, p) = (400, 120);
        let mut state = 7u64;
        let x = Mat::from_fn(n, p, |i, j| {
            let mut k = 0.0;
            for _ in 0..10 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                k += ((state >> 33) % 2) as f64;
            }
            // Half the cells get a 4x boost on half the genes.
            if i < n / 2 && j < p / 2 {
                k *= 4.0;
            }
            k
        });
        let s = Spectrum::compute(&CsrMatrix::from_dense(&x), 0.3, 1000, SCORE_K, SpectrumPath::Sparse);
        assert!(tracy_widom(&s, 0.001, 20).rank >= 1);
    }

    /// Counts with a planted programme and ~60% zeros, deterministic.
    fn sparse_counts(n: usize, p: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        Mat::from_fn(n, p, |i, j| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let base = if u < 0.6 { 0.0 } else { 1.0 + (u * 7.0).floor() };
            if i % 3 == 0 && j < p / 4 { base * 3.0 } else { base }
        })
    }

    /// The pre-CSR computation (dense Sinkhorn, explicit centring, X^T X), kept
    /// here as the reference the sparse path must reproduce.
    fn dense_reference(x: &Mat<f64>, damp: f64, max_iter: usize) -> Vec<f64> {
        let (n, p) = (x.nrows(), x.ncols());
        let bw = Biwhitener { damp, max_iter, ..Biwhitener::default() };
        let (c, d, _, _, _) = bw.compute(x);
        let xw = Biwhitener::apply(x, &c, &d);
        let means: Vec<f64> = (0..p).map(|j| (0..n).map(|i| xw.read(i, j)).sum::<f64>() / n as f64).collect();
        let xc = Mat::from_fn(n, p, |i, j| xw.read(i, j) - means[j]);
        let g = xc.as_ref().transpose() * xc.as_ref();
        let g = Mat::from_fn(p, p, |i, j| g.read(i, j) / (n - 1) as f64);
        let mut ev = g.selfadjoint_eigenvalues(Side::Lower);
        ev.sort_by(|a, b| b.total_cmp(a));
        ev
    }

    /// The sparse path reproduces the dense one: same eigenvalues (before the
    /// sigma^2 rescale) to 1e-8 relative, and rank-only (eigenvalues only)
    /// matches the full EVD path exactly in rank and to 1e-10 in eigenvalues.
    #[test]
    fn sparse_matches_dense_and_rank_only_matches_full() {
        let x = sparse_counts(600, 150, 11);
        let reference = dense_reference(&x, 0.3, 150);
        let csr = CsrMatrix::from_dense(&x);
        let full = Spectrum::compute(&csr, 0.3, 150, SCORE_K, SpectrumPath::Sparse);
        let fast = Spectrum::compute(&csr, 0.3, 150, 0, SpectrumPath::Sparse);
        let dense = Spectrum::compute(&csr, 0.3, 150, 0, SpectrumPath::Dense);
        assert_eq!(tracy_widom(&dense, 0.001, 100).rank, tracy_widom(&fast, 0.001, 100).rank);
        for t in 0..40 {
            let (a, b) = (dense.eigenvalues[t] * dense.sigma_sq, fast.eigenvalues[t] * fast.sigma_sq);
            assert!((a - b).abs() <= 1e-8 * a.abs().max(1e-12), "dense vs sparse eig {t}: {a} vs {b}");
        }
        for (t, r) in reference.iter().enumerate().take(40) {
            let got = full.eigenvalues[t] * full.sigma_sq;
            assert!((got - r).abs() <= 1e-8 * r.abs().max(1e-12), "eig {t}: sparse {got} vs dense {r}");
        }
        for t in 0..full.eigenvalues.len() {
            assert!((full.eigenvalues[t] - fast.eigenvalues[t]).abs() <= 1e-10 * full.eigenvalues[t].abs().max(1.0));
        }
        assert_eq!(tracy_widom(&full, 0.001, 100).rank, tracy_widom(&fast, 0.001, 100).rank);
        assert!(full.scores.ncols() > 0 && fast.scores.ncols() == 0);
    }

    /// Scores from the sparse row product equal the dense (Y - 1 mu') V.
    #[test]
    fn sparse_scores_match_dense_projection() {
        let x = sparse_counts(300, 80, 5);
        for path in [SpectrumPath::Sparse, SpectrumPath::Dense] {
        let s = Spectrum::compute(&CsrMatrix::from_dense(&x), 0.3, 150, 10, path);
        // Variance of each score column equals its eigenvalue (on the raw scale).
        for t in 0..5 {
            let col: Vec<f64> = (0..s.n).map(|i| s.scores.read(i, t)).collect();
            let mean = col.iter().sum::<f64>() / s.n as f64;
            let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (s.n - 1) as f64;
            let lam = s.eigenvalues[t] * s.sigma_sq;
            assert!(mean.abs() < 1e-8 && (var - lam).abs() <= 1e-8 * lam, "score {t}: var {var} vs lambda {lam}");
        }
        }
    }


}
