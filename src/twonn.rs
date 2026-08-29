//! TwoNN intrinsic-dimension estimator (Facco et al. 2017), following the
//! `intRinsic::twonn_mle` reference implementation.
//!
//! For each point take the distances to its first two neighbours and form
//! mu_i = r_2 / r_1. If the data lies on a d-dimensional manifold and the
//! density is locally constant, mu_i is Pareto(1, d), so
//!
//!   d_hat = (n - 1) / sum_i log(mu_i)
//!
//! with a Gamma(n, n-1) sampling law for the estimator, which gives the
//! confidence interval directly.
//!
//! This answers a *different* question from the spectral heuristics: the
//! dimension of the manifold the cells sit on, not the number of linear
//! components above the noise floor. It is normally much smaller, and it is a
//! useful upper bound on how many UMAP/diffusion coordinates could possibly
//! carry structure.

use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Gamma};

use crate::geom::Cloud;
use crate::rank::Estimate;

/// TwoNN MLE with the `intRinsic` trimming and confidence interval.
///
/// `c_trimmed` drops that top fraction of mu values before fitting (0.01 in
/// the reference). The trim matters: a single pair of near-duplicate cells
/// produces a huge mu and drags the estimate down.
///
pub fn two_nn(c: &Cloud, c_trimmed: f64, conf: f64) -> Estimate {
    let all: Vec<usize> = (0..c.len()).collect();
    estimate_from_mus(neighbours(c, &all).0, c_trimmed, conf)
}

/// mu_i = r_2 / r_1 for each point in `idx`, plus the mean r_2 -- the length
/// scale the estimate was taken at.
///
/// `idx` selects a subset of the cloud, which is what the scale analysis
/// decimates over; distances are read out of the cloud's Gram matrix rather
/// than recomputed per level.
pub(crate) fn neighbours(c: &Cloud, idx: &[usize]) -> (Vec<f64>, f64) {
    let pairs: Vec<(f64, f64)> = idx
        .par_iter()
        .filter_map(|&i| {
            let (mut r1, mut r2) = (f64::INFINITY, f64::INFINITY);
            for &j in idx {
                if j == i {
                    continue;
                }
                let d2 = c.d2(i, j);
                if d2 < r1 {
                    r2 = r1;
                    r1 = d2;
                } else if d2 < r2 {
                    r2 = d2;
                }
            }
            // Duplicate points give r1 = 0 and an infinite ratio; the
            // estimator has nothing to say about them, so drop them.
            (r1 > 0.0 && r2.is_finite()).then(|| ((r2 / r1).sqrt(), r2.sqrt()))
        })
        .collect();

    let mean_r2 = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len().max(1) as f64;
    (pairs.into_iter().map(|p| p.0).collect(), mean_r2)
}

/// The MLE, trimming and CI, separated out so it can be tested against the
/// closed-form Pareto case without building a matrix.
fn estimate_from_mus(mut mus: Vec<f64>, c_trimmed: f64, conf: f64) -> Estimate {
    let n_original = mus.len();
    mus.sort_by(f64::total_cmp);
    let keep = ((n_original as f64 * (1.0 - c_trimmed)).floor() as usize).max(1);
    mus.truncate(keep);

    let n = mus.len();
    let sum_log: f64 = mus.iter().map(|m| m.ln()).sum();
    if n < 2 || sum_log <= 0.0 {
        return Estimate {
            name: "twonn",
            rank: 0,
            detail: format!("undefined: {n} usable neighbour ratios"),
            stat: None,
            pvalues: Vec::new(),
        };
    }

    // Unbiased MLE, matching intRinsic's `unbiased = TRUE` default.
    let d = (n - 1) as f64 / sum_log;
    let a = (1.0 - conf) / 2.0;
    let g = Gamma::new(n as f64, (n - 1) as f64).expect("n >= 2");
    let (lo, hi) = (d * g.inverse_cdf(a), d * g.inverse_cdf(1.0 - a));

    Estimate {
        name: "twonn",
        rank: d.round() as usize,
        detail: format!(
            "d = {d:.2} ({:.0}% CI {lo:.2}-{hi:.2}), {n}/{n_original} ratios after {:.0}% trim",
            conf * 100.0,
            c_trimmed * 100.0
        ),
        stat: Some(d),
        pvalues: Vec::new(),
    }
}

/// One decimation level of the scale analysis.
pub struct ScalePoint {
    /// Points in each subsample at this level.
    pub n: usize,
    /// Mean d_hat over the replicate subsamples.
    pub d: f64,
    /// Observed min/max of d_hat across replicates -- an empirical spread, not
    /// a confidence interval. Degenerate at the top level, where there is only
    /// one possible subsample.
    pub spread: (f64, f64),
    /// Mean distance to the second neighbour: the length scale probed here.
    pub mean_r2: f64,
}

/// Below this a subsample has too few ratios for the MLE to mean anything.
const MIN_POINTS: usize = 50;

/// Consecutive decimation levels that must agree before it is called a plateau.
const MIN_PLATEAU_LEVELS: usize = 3;

/// Facco's scale analysis (Sci. Rep. 7, 12140, 2017, Fig. 3 and the surrounding
/// text): repeatedly halve the sample and watch how d_hat moves.
///
/// TwoNN uses only the first two neighbours, so it always probes the smallest
/// scale available. Removing points pushes the typical neighbour distance out
/// (as N^(-1/d)), which is the only knob for probing a larger one. At small
/// scale the estimate is inflated by noise, which is full-rank; at large scale
/// it is deflated by curvature and density variation. In between, if the data
/// really lies near a manifold, d_hat(N) is flat.
///
/// "The relevant ID of the dataset can be obtained by finding a range of N for
/// which d_hat(N) is constant, and thus a plateau in the graph of d_hat(N)."
///
/// Levels run from the full cloud down by halving until `MIN_POINTS`. `reps`
/// random subsamples are drawn per level (the top level has only one, so it is
/// drawn once whatever `reps` says). Cost is about 2x a single TwoNN fit: the
/// halved levels are quadratically cheaper and the series converges.
pub fn scale_analysis(c: &Cloud, reps: usize, c_trimmed: f64) -> Vec<ScalePoint> {
    let base: Vec<usize> = (0..c.len()).collect();
    let mut out = Vec::new();
    let mut m = base.len();
    let mut level = 0u64;
    while m >= MIN_POINTS {
        let reps = if m == base.len() { 1 } else { reps.max(1) };
        let mut ds = Vec::with_capacity(reps);
        let mut r2s = Vec::with_capacity(reps);
        for rep in 0..reps {
            // Deterministic per (level, rep): a diagnostic you cannot rerun is
            // not a diagnostic.
            let rows = sample(&base, m, level << 32 | rep as u64);
            let (mus, mean_r2) = neighbours(c, &rows);
            if let Some(d) = fit(mus, c_trimmed) {
                ds.push(d);
                r2s.push(mean_r2);
            }
        }
        if !ds.is_empty() {
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            out.push(ScalePoint {
                n: m,
                d: mean(&ds),
                spread: (
                    ds.iter().copied().fold(f64::INFINITY, f64::min),
                    ds.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ),
                mean_r2: mean(&r2s),
            });
        }
        m /= 2;
        level += 1;
    }
    out
}

/// Read the dimension off the plateau: the longest run of consecutive levels
/// whose estimates all lie within `tol` (relative) of each other.
///
/// The bound is on the *whole run*, not on adjacent steps. A steady decline of
/// 9% per level passes an adjacent-step test at tol = 0.10 while dropping by
/// half over five levels, and calling that a plateau is exactly the error this
/// is meant to avoid -- d_hat(N) has to be constant, not merely slow.
///
/// Three levels is the minimum that counts as "a range of N": two adjacent
/// estimates landing within tol of each other happens all the time on a
/// monotone curve.
///
/// ponytail: Facco reads the plateau off the plot by eye, and the printed table
/// is still the real deliverable. This is a convenience, not a replacement.
pub fn plateau(points: &[ScalePoint], tol: f64) -> Estimate {
    if points.len() < MIN_PLATEAU_LEVELS {
        return Estimate {
            name: "twonn-plateau",
            rank: 0,
            detail: format!(
                "no plateau: {} decimation levels, need {MIN_PLATEAU_LEVELS}",
                points.len()
            ),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    let ds: Vec<f64> = points.iter().map(|p| p.d).collect();
    let best = crate::rank::longest_flat_run(&ds, tol);
    let run = &points[best.start..=best.end];
    if run.len() < MIN_PLATEAU_LEVELS {
        let (h, l) = (&points[0], points.last().unwrap());
        return Estimate {
            name: "twonn-plateau",
            rank: 0,
            detail: format!(
                "no plateau: d drifts {:.1} -> {:.1} from N={} to N={}, no {MIN_PLATEAU_LEVELS} \
                 consecutive levels within {:.0}% -- read the table, not this row",
                h.d, l.d, h.n, l.n, tol * 100.0
            ),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    let d = run.iter().map(|p| p.d).sum::<f64>() / run.len() as f64;
    Estimate {
        name: "twonn-plateau",
        rank: d.round() as usize,
        detail: format!(
            "d = {d:.2} over N = {}..{} ({} levels within {:.0}%)",
            run.last().unwrap().n,
            run[0].n,
            run.len(),
            tol * 100.0
        ),
        stat: Some(d),
        pvalues: Vec::new(),
    }
}

/// Trimmed unbiased MLE, without the CI. `None` when there is nothing to fit.
fn fit(mut mus: Vec<f64>, c_trimmed: f64) -> Option<f64> {
    mus.sort_by(f64::total_cmp);
    mus.truncate(((mus.len() as f64 * (1.0 - c_trimmed)).floor() as usize).max(1));
    let sum_log: f64 = mus.iter().map(|m| m.ln()).sum();
    (mus.len() >= 2 && sum_log > 0.0).then(|| (mus.len() - 1) as f64 / sum_log)
}

/// `k` distinct entries of `pool`, by partial Fisher-Yates with a splitmix64
/// stream. Returned sorted, which keeps the Gram build cache-friendly.
fn sample(pool: &[usize], k: usize, seed: u64) -> Vec<usize> {
    if k >= pool.len() {
        return pool.to_vec();
    }
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut v = pool.to_vec();
    for i in 0..k {
        let j = i + (next() % (v.len() - i) as u64) as usize;
        v.swap(i, j);
    }
    v.truncate(k);
    v.sort_unstable();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Exact Pareto(1, d) ratios must return d. This checks the estimator
    /// itself, with the geometry taken out of the picture: the inverse CDF of
    /// Pareto(1,d) at u is (1-u)^(-1/d).
    #[test]
    fn recovers_pareto_shape() {
        for d in [2.0, 5.0, 11.0] {
            let n = 20000;
            let mus: Vec<f64> = (1..=n)
                .map(|i| {
                    let u = i as f64 / (n as f64 + 1.0);
                    (1.0 - u).powf(-1.0 / d)
                })
                .collect();
            let est = estimate_from_mus(mus, 0.0, 0.95);
            let got: f64 = est.detail[4..].split_whitespace().next().unwrap().parse().unwrap();
            assert!((got - d).abs() < 0.15, "d={d}: {}", est.detail);
        }
    }

    /// Points drawn on a 3-dimensional linear subspace of R^20 have intrinsic
    /// dimension 3, whatever the ambient dimension says.
    #[test]
    fn recovers_linear_subspace_dimension() {
        let (n, ambient, intrinsic) = (2000, 20, 3);
        let mut state = 12345u64;
        let mut rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // latent (n x 3), then a fixed random embedding into R^20.
        let latent: Vec<f64> = (0..n * intrinsic).map(|_| rand()).collect();
        let basis: Vec<f64> = (0..intrinsic * ambient).map(|_| rand() - 0.5).collect();
        let x = Mat::from_fn(n, ambient, |i, j| {
            (0..intrinsic)
                .map(|k| latent[i * intrinsic + k] * basis[k * ambient + j])
                .sum()
        });
        let est = two_nn(&Cloud::new(&x, 2000), 0.01, 0.95);
        assert_eq!(est.rank, intrinsic, "{}", est.detail);
    }

    /// On a clean manifold there is nothing for the scale analysis to reveal:
    /// d_hat must stay flat across decimation levels and the plateau must find
    /// the true dimension.
    #[test]
    fn plateau_is_flat_on_a_clean_manifold() {
        let (n, ambient, intrinsic) = (1600, 20, 3);
        let x = embedded_uniform(n, ambient, intrinsic, 0.0, 999);
        let pts = scale_analysis(&Cloud::new(&x, 1600), 3, 0.01);
        assert!(pts.len() >= 4, "only {} levels", pts.len());
        for p in &pts {
            assert!((p.d - 3.0).abs() < 0.6, "N={} gave d={:.2}", p.n, p.d);
        }
        assert_eq!(plateau(&pts, 0.10).rank, intrinsic);
    }

    /// The point of the whole exercise: full-rank noise inflates TwoNN at the
    /// smallest scale, and decimating must walk the estimate back down towards
    /// the true manifold dimension.
    #[test]
    fn decimation_deflates_a_noise_inflated_estimate() {
        let (n, ambient, intrinsic) = (1600, 20, 3);
        let x = embedded_uniform(n, ambient, intrinsic, 0.02, 4242);
        let pts = scale_analysis(&Cloud::new(&x, 1600), 3, 0.01);
        let (first, last) = (pts[0].d, pts.last().unwrap().d);
        assert!(first > 3.5, "noise did not inflate the estimate: {first:.2}");
        assert!(last < first, "decimation did not deflate: {first:.2} -> {last:.2}");
    }

    /// Too few points to decimate is not a plateau, and must not be a panic:
    /// a file with under 50 cells reaches here with an empty level list.
    #[test]
    fn too_few_levels_is_not_a_plateau() {
        assert_eq!(plateau(&[], 0.10).rank, 0);
    }

    /// Uniform points on an `intrinsic`-dimensional linear subspace of
    /// R^ambient, plus isotropic full-rank jitter.
    fn embedded_uniform(n: usize, ambient: usize, intrinsic: usize, noise: f64, seed: u64) -> Mat<f64> {
        let mut state = seed;
        let mut rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let latent: Vec<f64> = (0..n * intrinsic).map(|_| rand()).collect();
        let basis: Vec<f64> = (0..intrinsic * ambient).map(|_| rand() - 0.5).collect();
        let jitter: Vec<f64> = (0..n * ambient).map(|_| noise * (rand() - 0.5)).collect();
        Mat::from_fn(n, ambient, |i, j| {
            (0..intrinsic)
                .map(|k| latent[i * intrinsic + k] * basis[k * ambient + j])
                .sum::<f64>()
                + jitter[i * ambient + j]
        })
    }
}
