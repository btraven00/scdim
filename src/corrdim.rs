//! Grassberger-Procaccia correlation dimension.
//!
//! The correlation integral counts pairs closer than r:
//!
//!   C(r) = 2 / (N(N-1)) * #{ i < j : ||x_i - x_j|| < r }
//!
//! On a D-dimensional set C(r) ~ r^D, so D is the slope of log C against log r.
//! Unlike TwoNN, which fits a parametric law to one neighbour ratio, this reads
//! the scaling of the whole pair-distance distribution -- which is what makes it
//! the better tool for continuous, branching structure (a differentiation
//! trajectory) as opposed to a set of discrete clusters.
//!
//! Nothing about it is a fit to clusters, so it will not return "the number of
//! cell types". It returns the dimension of the set the cells trace out.
//!
//! ## The sample-size ceiling
//!
//! This is the estimator's hard limitation and it is not a detail. Estimating
//! a dimension D from N points needs, roughly,
//!
//!   D <= 2 log10(N)
//!
//! (Eckmann & Ruelle 1992). With 2000 points that is D <= 6.6. Above the
//! ceiling the correlation integral simply has no scaling range left to
//! measure, and the estimator returns something far too small rather than
//! failing loudly -- so [`correlation_dimension`] reports the ceiling next to
//! every estimate, and you should read the two together.

use crate::geom::Cloud;
use crate::rank::{longest_flat_run, Estimate};

/// One point of the log C / log r curve.
pub struct GpPoint {
    pub r: f64,
    /// Fraction of pairs closer than `r`.
    pub c: f64,
    /// Local slope d log C / d log r, i.e. the dimension estimate at this
    /// scale. NaN at the last point, which has no successor.
    pub slope: f64,
}

/// Correlation integral and its local slope, over `bins` scales. Returns the
/// curve and the number of points it was built from -- the caller needs that
/// for the Eckmann-Ruelle ceiling, and it is not the cap the cloud was built
/// with (the stride through the rows rarely divides evenly).
///
/// Scales are chosen so that C is log-spaced from 1e-4 to 0.3: every slope
/// estimate then rests on a comparable number of pairs, and none of them sit in
/// the saturated region where C -> 1 and the slope collapses to zero by
/// construction.
pub fn correlation_curve(c: &Cloud, bins: usize) -> (Vec<GpPoint>, usize) {
    let m = c.len();
    let mut d = pair_distances(c);
    if d.len() < 100 {
        return (Vec::new(), m);
    }
    d.sort_by(f64::total_cmp);
    let n = d.len() as f64;

    let mut curve: Vec<GpPoint> = (0..bins)
        .filter_map(|k| {
            let t = k as f64 / (bins - 1).max(1) as f64;
            let c = 1e-4_f64.powf(1.0 - t) * 0.3_f64.powf(t); // log-spaced 1e-4 -> 0.3
            let idx = ((c * n).round() as usize).clamp(1, d.len() - 1);
            (d[idx] > 0.0).then(|| GpPoint {
                r: d[idx],
                c: idx as f64 / n,
                slope: f64::NAN,
            })
        })
        .collect();
    for i in 0..curve.len().saturating_sub(1) {
        let dr = curve[i + 1].r.ln() - curve[i].r.ln();
        if dr > 0.0 {
            curve[i].slope = (curve[i + 1].c.ln() - curve[i].c.ln()) / dr;
        }
    }
    (curve, m)
}

/// Read the dimension off the flat part of the local-slope curve.
///
/// `n_points` is the sample size the curve was built from; it sets the
/// Eckmann-Ruelle ceiling reported alongside the estimate.
pub fn correlation_dimension(curve: &[GpPoint], n_points: usize, tol: f64) -> Estimate {
    let ceiling = 2.0 * (n_points as f64).log10();
    let slopes: Vec<f64> = curve
        .iter()
        .map(|p| p.slope)
        .filter(|s| s.is_finite())
        .collect();
    if slopes.len() < 3 {
        return Estimate {
            name: "corr-dim",
            rank: 0,
            detail: "not enough scales with pairs in them".to_string(),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    let run = longest_flat_run(&slopes, tol);
    if run.end - run.start < 2 {
        return Estimate {
            name: "corr-dim",
            rank: 0,
            detail: format!(
                "no scaling range: local slope runs {:.1} -> {:.1} with no 3 consecutive \
                 scales within {:.0}% -- read the table, not this row (ceiling {ceiling:.1})",
                slopes[0],
                slopes[slopes.len() - 1],
                tol * 100.0
            ),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    let d = slopes[run.start..=run.end].iter().sum::<f64>() / (run.end - run.start + 1) as f64;
    let warn = if d > 0.8 * ceiling {
        format!(
            " -- AT THE CEILING: {n_points} points support D <= {ceiling:.1} \
             (Eckmann-Ruelle), so this is a lower bound, not a measurement"
        )
    } else {
        format!(" (ceiling {ceiling:.1})")
    };
    Estimate {
        name: "corr-dim",
        rank: d.round() as usize,
        detail: format!(
            "D = {d:.2} over {} scales flat to {:.0}%{warn}",
            run.end - run.start + 1,
            tol * 100.0
        ),
        stat: Some(d),
        pvalues: Vec::new(),
    }
}

/// All m(m-1)/2 pairwise distances.
fn pair_distances(c: &Cloud) -> Vec<f64> {
    let m = c.len();
    let mut out = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            out.push(c.dist(i, j));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A filled 3-cube embedded in R^20 has correlation dimension 3. This is
    /// below the Eckmann-Ruelle ceiling for 2000 points, so it is a fair test.
    #[test]
    fn recovers_cube_dimension() {
        let (n, ambient, intrinsic) = (2000, 20, 3);
        let mut r = lcg(31337);
        let latent: Vec<f64> = (0..n * intrinsic).map(|_| r()).collect();
        let basis: Vec<f64> = (0..intrinsic * ambient).map(|_| r() - 0.5).collect();
        let x = Mat::from_fn(n, ambient, |i, j| {
            (0..intrinsic)
                .map(|k| latent[i * intrinsic + k] * basis[k * ambient + j])
                .sum()
        });
        let (curve, m) = correlation_curve(&Cloud::new(&x, n), 20);
        let est = correlation_dimension(&curve, m, 0.15);
        assert_eq!(est.rank, intrinsic, "{}", est.detail);
    }

    /// A one-dimensional curve stays one-dimensional however it is embedded --
    /// the case the estimator exists for (a trajectory, not clusters).
    #[test]
    fn recovers_curve_dimension() {
        let n = 2000;
        let x = Mat::from_fn(n, 10, |i, j| {
            let t = i as f64 / n as f64 * 6.0;
            match j {
                0 => t.cos(),
                1 => t.sin(),
                2 => t / 6.0,
                _ => 0.1 * (t * (j as f64)).sin(),
            }
        });
        let (curve, m) = correlation_curve(&Cloud::new(&x, n), 20);
        let est = correlation_dimension(&curve, m, 0.15);
        assert_eq!(est.rank, 1, "{}", est.detail);
    }

    /// The correlation integral must be monotone in r and land in (0, 1].
    #[test]
    fn curve_is_monotone() {
        let mut r = lcg(7);
        let x = Mat::from_fn(500, 5, |_, _| r());
        let (curve, _) = correlation_curve(&Cloud::new(&x, 500), 15);
        assert!(curve.len() > 5);
        for w in curve.windows(2) {
            assert!(w[1].r >= w[0].r && w[1].c > w[0].c);
            assert!(w[0].c > 0.0 && w[0].c <= 1.0);
        }
    }
}
