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
use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::rmt::RmtTheory;

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
    /// PCA scores, n x min(SCORE_K, rank): the cells in the leading principal
    /// subspace, on the U*Sigma scale.
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
    /// Biwhiten, centre, and diagonalise. O(min(n,p)^3).
    ///
    /// `damp` and `max_iter` are the Sinkhorn-Knopp knobs. Real count data
    /// oscillates under the undamped step; if `biwhitening_converged` comes
    /// back false, lower `damp` or raise `max_iter` before believing anything.
    pub fn compute(x: &Mat<f64>, damp: f64, max_iter: usize) -> Self {
        let (n, p) = (x.nrows(), x.ncols());
        let q = p as f64 / n as f64;

        // ponytail: Sinkhorn is ~70% of the run at 20k cells (12.1s of 17.1s)
        // and this path is dense. ../scwarp `feat/biwhitening` has a sparse CSR
        // Sinkhorn-Knopp (`scwarp_core::biwhiten::sinkhorn_knopp`, CPU + wgpu +
        // CUDA) plus its own MP edge/rank/KS diagnostics -- swap to it when that
        // branch lands on main. Three caveats measured before switching:
        //   1. GPU is not the lever. scwarp's own profile has the dense GPU
        //      kernel losing to their CPU sparse loop at every size, and the
        //      sparse GPU winning only ~14% at 20k x 2k once setup is counted.
        //   2. Sparsity is not the lever *here*. Top-variance gene selection
        //      keeps the dense genes: the matrix fed to Sinkhorn is 49.6% dense,
        //      where sparse and dense cost about the same. It becomes the lever
        //      only if whitening moves to the full gene set (~7%), which is what
        //      scwarp argues for anyway.
        //   3. The real lever is the iteration count. scwarp uses Landa et al.'s
        //      plain Sinkhorn on the variance matrix and converges in 20-35
        //      iterations; this Chardes variant oscillates under damping and
        //      needs ~225. Two further disagreements to settle first: scwarp
        //      does not mean-centre after whitening (says it shifts the spectrum
        //      off the MP law) and whitens the whole gene set rather than an HVG
        //      subset.
        let bw = Biwhitener {
            damp,
            max_iter,
            ..Biwhitener::default()
        };
        let (c, d, _, converged, residual) = bw.compute(x);
        let xw = Biwhitener::apply(x, &c, &d);

        // S is a covariance only for zero-mean data; centre after biwhitening
        // so the Sinkhorn stage stays in the non-negative domain.
        let means: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| xw.read(i, j)).sum::<f64>() / n as f64)
            .collect();
        let xc = Mat::from_fn(n, p, |i, j| xw.read(i, j) - means[j]);

        // X^T X and X X^T share their nonzero eigenvalues; take the smaller.
        let m = n.min(p);
        let denom = (n - 1) as f64;
        let gram = if p <= n {
            xc.as_ref().transpose() * xc.as_ref()
        } else {
            xc.as_ref() * xc.as_ref().transpose()
        };

        let gram = Mat::from_fn(m, m, |i, j| gram.read(i, j) / denom);

        let evd = SelfAdjointEigendecomposition::new(gram.as_ref(), Side::Lower);
        let mut eigenvalues: Vec<f64> = (0..m).map(|i| evd.s().column_vector().read(i)).collect();
        eigenvalues.reverse(); // faer returns ascending

        // Scores = U*Sigma either way. When the Gram is X^T X the eigenvectors
        // are gene loadings V and scores = Xc*V; when it is X X^T they are the
        // cell directions U and scores = U*sqrt((n-1)*lambda).
        let k = SCORE_K.min(m);
        let u = evd.u();
        let col = |t: usize| m - 1 - t; // t-th largest
        let scores = if p <= n {
            let v = Mat::from_fn(p, k, |i, t| u.read(i, col(t)));
            xc.as_ref() * v.as_ref()
        } else {
            Mat::from_fn(n, k, |i, t| {
                u.read(i, col(t)) * (denom * eigenvalues[t].max(0.0)).sqrt()
            })
        };

        // sigma^2 by median-matching the bulk against MP: robust to the
        // outliers we are about to count, unlike a trace-based estimate.
        let theory = RmtTheory { q };
        let lp = theory.lambda_plus();
        let mut bulk: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&e| e > 1e-3 && e <= lp * eigenvalues[0].max(1.0))
            .collect();
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
            scores,
        }
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
        let s = Spectrum::compute(&x, 0.3, 1000);
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
        let s = Spectrum::compute(&x, 0.3, 1000);
        assert!(tracy_widom(&s, 0.001, 20).rank >= 1);
    }


}
