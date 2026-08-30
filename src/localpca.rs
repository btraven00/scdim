//! Local PCA: the same covariance, read at growing radius.
//!
//! `tracy-widom` is the rank of the covariance of the whole cloud;
//! `twonn` is the dimension of the tangent space at a point. They are the same
//! quantity at two scales -- the intrinsic dimension is the rank of the *local*
//! covariance as r -> 0 -- so eigendecomposing each cell's k-neighbourhood and
//! growing k walks continuously from one row to the other.
//!
//! In a ball of radius r on a d-dimensional manifold the local
//! covariance has d eigenvalues of order r^2 (the tangent directions) and up to
//! d(d+1)/2 of order r^4 (the second fundamental form), with higher orders
//! below that. A flat patch gives the same d at every k; a curved one spends
//! extra linear dimensions as the ball grows. The gap between the two existing
//! rows is not error, it is bending.
//!
//! Little, Maggioni & Rosasco, ACHA 43 (2017) -- multiscale SVD.
//!
//! ## Read the minimum, not the ends
//!
//! The curve is the same three regimes `corrdim` documents for C(r), for the
//! same reason. At small k the ball is inside the noise, which is full-rank, so
//! d is inflated. At large k it swallows curvature and neighbouring cell types,
//! so d is inflated again. The manifold is the dip in between, and that is the
//! number the row reports -- the ends are the failure modes, not the answer.
//!
//! ## No threshold
//!
//! Counting "eigenvalues above the r^4 shoulder" needs a cut, and every cut
//! here would be one more knob calibrated on one dataset. The participation
//! ratio of the local spectrum needs none:
//!
//!   d = (sum lambda)^2 / sum lambda^2
//!
//! It is exactly d for an isotropic d-dimensional patch and moves smoothly,
//! which is what makes the curve over k readable rather than a staircase of
//! threshold artefacts. Its bias is worth knowing: being a ratio of moments it
//! weights by variance, so it counts directions carrying *comparable* variance
//! and is nearly blind to a single weak one. A 90-degree arc of a circle has a
//! second eigenvalue 4% of its first and still reads d = 1.09. So this is a
//! conservative estimator -- it will not manufacture dimensions out of mild
//! curvature, and it will not resolve them either.
//!
//! It does need one correction. A covariance estimated from k+1 points has
//! dispersed sample eigenvalues even when the population ones are equal, which
//! deflates the ratio: for Wishart, `E[tr S] = d` and
//! `E[tr S^2] = d(d+k+1)/k`, so the observed value converges to `d / (1 + d/k)`
//! rather than to d -- 2.2 instead of 3 at k = 8, measured. Inverting that
//! gives `d = PR / (1 - PR/k)`, which is the whole correction. Without it the
//! small-k end is biased low, and the small-k end is the intrinsic dimension:
//! the correction is largest exactly where the number matters most.
//!
//! ## The ceiling
//!
//! k+1 points span at most k dimensions, and the cloud lives in `embed_dim`,
//! so no level can report more than min(k, embed_dim) however curved the data
//! is. Same shape of limitation as Eckmann-Ruelle in `corrdim`, and it is
//! printed next to every level for the same reason: above it the estimator
//! returns something too small without failing.

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Mat, Side};
use rayon::prelude::*;

use crate::geom::Cloud;
use crate::rank::Estimate;

/// Neighbourhoods eigendecomposed per level, at k = `K_REF` and below. 256
/// centres pin the mean spectrum well enough that spending 2000 EVDs on a
/// smoother one would be paying for decimals nobody reads.
const MAX_CENTRES: usize = 256;
const K_REF: usize = 256;

/// Floor on the centre count at the largest radii. A k = 1024 neighbourhood is
/// an eighth of an 8000-cell cloud, so 16 of them already overlap heavily --
/// the variance that matters up there is between spectra, not between centres.
const MIN_CENTRES: usize = 16;

/// Smallest and largest neighbourhood, doubling in between. Below 16 the
/// ceiling makes the level uninformative; above m/4 the "neighbourhood" is a
/// quarter of the cloud and the answer is just the global rank again.
///
/// The cap was 256, and measured against clouds of 2000, 4000 and 8000 cells
/// the dip landed on that last rung every single time -- the row was reporting
/// a constant of mine rather than a property of the data. Reaching 1024 costs
/// little extra because the centre count falls as 1/k^2 against an EVD that
/// costs k^3, so each doubling of k merely doubles the level's cost instead of
/// octupling it.
const K_MIN: usize = 16;
const K_CAP: usize = 1024;

/// One radius of the walk.
pub struct LocalPoint {
    /// Neighbours per local PCA.
    pub k: usize,
    /// Neighbourhoods averaged at this radius.
    pub centres: usize,
    /// Mean participation ratio of the local spectrum: the dimension at this
    /// radius.
    pub d: f64,
    /// Mean distance to the k-th neighbour -- the radius actually probed.
    pub mean_r: f64,
    /// `min(k, embed_dim)`: no level can report more than this.
    pub ceiling: usize,
    /// 10th and 90th percentile of the per-neighbourhood dimension.
    ///
    /// `d` is computed from the *mean* normalised spectrum, which is the right
    /// estimator but says nothing about whether the cloud has one dimension
    /// everywhere. A manifold of constant dimension gives a tight spread; a
    /// cloud with a 1-D filament joined to a 3-D body gives the same mean and a
    /// wide one, and only this column separates them.
    pub spread: (f64, f64),
}

/// Local dimension against neighbourhood size.
///
/// Builds its own k-NN lists at the largest level and slices the prefixes --
/// the lists are sorted, so the first k of the 128 nearest *are* the k
/// nearest. One extra O(m^2) scan, which is far from the bottleneck (every
/// other geometric diagnostic here is O(m^2) or O(m^3)).
pub fn local_pca(c: &Cloud, embed_dim: usize) -> Vec<LocalPoint> {
    let m = c.len();
    let k_max = K_CAP.min(m / 4);
    if k_max < K_MIN {
        return Vec::new();
    }
    let nbr = c.knn(k_max);
    if nbr.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut k = K_MIN;
    while k <= k_max {
        // Centres as 1/k^2 against an EVD that costs k^3: cost per level then
        // grows linearly in k rather than cubically, and the ladder can reach
        // a radius where the dip has somewhere interior to land.
        let want = (MAX_CENTRES * K_REF * K_REF / (k * k)).clamp(MIN_CENTRES, MAX_CENTRES);
        let stride = m.div_ceil(want).max(1);
        let centres: Vec<usize> = (0..m).step_by(stride).collect();
        let per_centre: Vec<(Vec<f64>, f64)> = centres
            .par_iter()
            .filter_map(|&i| {
                let mut sup = Vec::with_capacity(k + 1);
                sup.push(i);
                sup.extend(nbr[i][..k].iter().map(|&(_, j)| j));
                normalised_spectrum(c, &sup).map(|s| (s, nbr[i][k - 1].0.sqrt()))
            })
            .collect();
        if !per_centre.is_empty() {
            let n = per_centre.len() as f64;
            let mut mean = vec![0.0f64; k + 1];
            for (s, _) in &per_centre {
                for (acc, v) in mean.iter_mut().zip(s) {
                    *acc += v / n;
                }
            }
            // Each spectrum sums to 1, so the mean does too: PR = 1 / sum l^2.
            // Then undo the Wishart deflation, and clamp: as PR approaches k
            // the correction diverges, which is the ceiling asserting itself.
            let pr = 1.0 / mean.iter().map(|l| l * l).sum::<f64>();
            let ceiling = k.min(embed_dim);
            // Same debias and clamp as the headline, applied per neighbourhood.
            let debias = |p: f64| (p / (1.0 - p / k as f64)).clamp(1.0, ceiling as f64);
            let mut each: Vec<f64> = per_centre
                .iter()
                .map(|(sp, _)| debias(1.0 / sp.iter().map(|l| l * l).sum::<f64>()))
                .collect();
            each.sort_by(f64::total_cmp);
            let q = |f: f64| each[((f * each.len() as f64) as usize).min(each.len() - 1)];
            out.push(LocalPoint {
                k,
                centres: per_centre.len(),
                d: (pr / (1.0 - pr / k as f64)).clamp(1.0, ceiling as f64),
                mean_r: per_centre.iter().map(|p| p.1).sum::<f64>() / n,
                ceiling,
                spread: (q(0.1), q(0.9)),
            });
        }
        k *= 2;
    }
    out
}

/// Eigenvalues of the local covariance, descending, normalised to sum 1.
///
/// Read off the double-centred squared-distance matrix rather than the
/// coordinates -- classical MDS, the same "X^T X and X X^T share their nonzero
/// eigenvalues" identity `rank` uses. The neighbourhood has k+1 points and the
/// embedding has more dimensions than that, so this is the smaller of the two
/// and it needs nothing from the cloud but `d2`.
///
/// Normalising per point before averaging is load-bearing: a dense cell type
/// and a sparse one have local spectra differing by orders of magnitude, and a
/// raw mean would be the sparse regions' answer alone.
fn normalised_spectrum(c: &Cloud, sup: &[usize]) -> Option<Vec<f64>> {
    let n = sup.len();
    let d2 = Mat::from_fn(n, n, |a, b| c.d2(sup[a], sup[b]));
    let row: Vec<f64> = (0..n)
        .map(|a| (0..n).map(|b| d2.read(a, b)).sum::<f64>() / n as f64)
        .collect();
    let grand = row.iter().sum::<f64>() / n as f64;
    let b = Mat::from_fn(n, n, |i, j| -0.5 * (d2.read(i, j) - row[i] - row[j] + grand));

    let evd = SelfAdjointEigendecomposition::new(b.as_ref(), Side::Lower);
    // Gram matrices are PSD in exact arithmetic and slightly indefinite in f64.
    let mut lam: Vec<f64> = (0..n)
        .map(|i| evd.s().column_vector().read(i).max(0.0))
        .collect();
    lam.reverse();
    let t: f64 = lam.iter().sum();
    if !(t > 0.0) {
        return None; // coincident points: nothing to decompose
    }
    for l in &mut lam {
        *l /= t;
    }
    Some(lam)
}

/// The walk as one row: the dip, and both ends so the shape is visible.
pub fn summary(levels: &[LocalPoint], embed_dim: usize) -> Estimate {
    let (Some(first), Some(last)) = (levels.first(), levels.last()) else {
        return Estimate {
            name: "local-pca",
            rank: 0,
            detail: "too few points for a local neighbourhood".to_string(),
            stat: None,
            pvalues: Vec::new(),
        };
    };
    if levels.len() < 3 {
        return Estimate {
            name: "local-pca",
            rank: first.d.round() as usize,
            detail: format!(
                "d = {:.2} at k={}: too few radii to find a dip, read the table",
                first.d, first.k
            ),
            stat: None,
            pvalues: Vec::new(),
        };
    }
    // A level pinned against its own ceiling is not a measurement, and it is
    // pinned *low*, so leaving it in makes it the minimum by construction and
    // the dip search reports the ceiling back to you.
    let measured: Vec<&LocalPoint> = levels
        .iter()
        .filter(|p| p.d < 0.95 * p.ceiling as f64)
        .collect();
    let Some(&dip) = measured.iter().min_by(|a, b| a.d.total_cmp(&b.d)) else {
        return Estimate {
            name: "local-pca",
            rank: 0,
            detail: format!(
                "every radius is at its ceiling (k = {}..{}, embedding {embed_dim}D): the \
                 neighbourhoods are too small to hold the dimension -- read the table",
                first.k, last.k
            ),
            stat: None,
            pvalues: Vec::new(),
        };
    };
    // A minimum at either end is not a dip: the curve is monotone over every
    // radius the ladder reaches, so the manifold regime lies outside it. Which
    // end says which constraint binds -- the noise floor is a property of the
    // data, the largest radius is `K_CAP` and `m/4`, not `--geom-cells`.
    let (lo, hi) = (measured[0], measured[measured.len() - 1]);
    let monotone = if measured.len() < 3 {
        " -- on too few unpinned radii to call a dip".to_string()
    } else if dip.k == lo.k {
        " -- but it is still rising at every larger radius, so the walk starts \
         above the manifold: an upper bound"
            .to_string()
    } else if dip.k == hi.k {
        format!(
            " -- but it is still falling at k={}, the largest radius the ladder \
             reaches: every scale here is noise-dominated, so this is an upper bound",
            hi.k
        )
    } else {
        String::new()
    };
    let ceil = if dip.d > 0.8 * dip.ceiling as f64 {
        format!(
            " -- AT THE CEILING: a {}-point neighbourhood in {embed_dim}D supports \
             d <= {}",
            dip.k + 1,
            dip.ceiling
        )
    } else {
        String::new()
    };
    Estimate {
        name: "local-pca",
        rank: dip.d.round() as usize,
        detail: format!(
            "d = {:.2} at the dip (k={}, r={:.2}); {:.2} at k={} -> {:.2} at k={}, \
             global rank {embed_dim}{ceil}{monotone}",
            dip.d, dip.k, dip.mean_r, first.d, first.k, last.d, last.k
        ),
        stat: Some(dip.d),
        pvalues: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The whole claim, in one test. A flat 3-dimensional patch has nothing to
    /// spend extra dimensions on, so d must sit at 3 at *every* radius -- which
    /// is what pins the debias, the double-centring and the normalisation, none
    /// of which are right for any other constant. Add full-rank noise and the
    /// smallest radius must inflate above it, with the dip still finding 3:
    /// that is the three-regime shape the row exists to show.
    #[test]
    fn flat_stays_put_and_noise_inflates_the_small_radius() {
        let (n, ambient, intrinsic) = (1200, 20, 3);
        let mut r = lcg(31337);
        let latent: Vec<f64> = (0..n * intrinsic).map(|_| r()).collect();
        let basis: Vec<f64> = (0..intrinsic * ambient).map(|_| r() - 0.5).collect();
        let flat = Mat::from_fn(n, ambient, |i, j| {
            (0..intrinsic)
                .map(|k| latent[i * intrinsic + k] * basis[k * ambient + j])
                .sum()
        });
        let levels = local_pca(&Cloud::new(&flat, n), ambient);
        assert!(levels.len() >= 3, "only {} levels", levels.len());
        for p in &levels {
            assert!(
                (p.d - 3.0).abs() < 0.4,
                "flat patch gave d = {:.2} at k = {}",
                p.d,
                p.k
            );
        }

        assert!(
            levels.windows(2).all(|w| w[1].mean_r > w[0].mean_r),
            "radius must grow with k"
        );

        // The same patch under isotropic full-rank jitter. The smallest ball
        // sits inside the noise, which has no preferred directions, so it must
        // read high; the dip must still be the manifold.
        let noisy = Mat::from_fn(n, ambient, |i, j| {
            flat.read(i, j) + 0.08 * (r() - 0.5)
        });
        let levels = local_pca(&Cloud::new(&noisy, n), ambient);
        let dip = levels.iter().map(|p| p.d).fold(f64::INFINITY, f64::min);
        assert!(
            levels[0].d > dip + 1.0,
            "noise did not inflate the smallest radius: {:.2} vs dip {dip:.2}",
            levels[0].d
        );
        assert!((dip - 3.0).abs() < 0.6, "dip missed the manifold: {dip:.2}");
    }
}
