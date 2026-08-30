//! Do two regions of the cloud vary along the same directions?
//!
//! `local-pca` measures how many directions a cell can move in. This asks a
//! different question with the same neighbourhoods: whether the directions
//! available *here* are the directions available *there*.
//!
//! Local PCA at a point gives an orthonormal basis `U_x` for its tangent space.
//! The singular values of `U_x^T U_y` are the cosines of the principal angles
//! between two such spaces, and the scalar worth reporting is
//!
//! ```text
//! overlap(x, y) = tr(P_x P_y) = ||U_x^T U_y||_F^2 = sum cos^2(theta_i)
//! ```
//!
//! which is the *effective number of shared dimensions*: exactly 3 for two
//! subspaces sharing a 3-plane and otherwise orthogonal, and continuous in
//! between. It is the Grassmannian chordal inner product, so it does not depend
//! on which basis was picked for either space.
//!
//! Biologically: overlapping tangent spaces mean two regions vary along the
//! same gene programs. Because this all happens in PCA-score space, a shared
//! direction maps back through the loadings to named genes.
//!
//! ## This is a diagnostic for continua, not for clusters
//!
//! If the data has well-separated clusters, `betti0` and `fiedler` say so and
//! the regions are already obvious -- per-cluster PCA answers the question and
//! this adds nothing. What it is for is the case those rows call one connected
//! piece, which so far is every dataset tried: there are no clusters, yet the
//! tangent space still rotates as you move, and nothing else here can see that.
//! A flat blob keeps one tangent space everywhere and gives a flat curve; a
//! curved manifold decays. **Tangent rotation is curvature, not clustering.**
//!
//! ## Two corrections without which this reads noise
//!
//! **Random subspaces already overlap.** For independent uniformly random
//! d-dimensional subspaces of R^D, `E[tr(P_x P_y)] = d^2/D` -- at d = 11 in a
//! 65-dimensional embedding that is 1.86 shared dimensions before any biology.
//! The excess over that null is the number to read, and it is reported next to
//! the raw value rather than instead of it.
//!
//! **Nearby neighbourhoods share their points.** At one or two hops the two
//! k-neighbourhoods are largely the same cells, so a high overlap there is
//! arithmetic, not evidence. The informative part of the curve is its far end:
//! the floor is how many directions genuinely survive across the whole cloud.
//!
//! There is also a ceiling, in the same spirit as Eckmann-Ruelle in `corrdim`.
//! The null is `d^2/D`, so the measurement only has room when `d/D` is small.
//! At d = 14 in a 29-dimensional embedding the null is 6.8 of 14 -- half the
//! answer is chance and there is little left to detect. The row prints the null
//! next to every reading, and shouts when it exceeds 30% of d.

use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{Mat, Side};
use rayon::prelude::*;

use crate::geom::Cloud;

/// Tangent spaces compared. Quadratic in this, and the curve is an average, so
/// there is no point buying decimals nobody reads.
const MAX_CENTRES: usize = 256;

/// Hop bins; the last one is a `>=` bucket.
const MAX_HOPS: usize = 10;

/// Above this fraction of `d`, the random-subspace null leaves too little room
/// for the measurement to say much.
pub const NULL_WARN: f64 = 0.3;

/// One distance bin of the overlap curve.
pub struct OverlapPoint {
    /// Graph distance in k-NN hops. `hops == MAX_HOPS` means "at least this".
    pub hops: usize,
    pub pairs: usize,
    /// Mean `tr(P_x P_y)`: shared dimensions, including the chance baseline.
    pub shared: f64,
    /// The same, less the random-subspace null.
    pub excess: f64,
}

/// Tangent-space overlap against graph distance.
///
/// `d` is the tangent dimension to use, the same for every point so the numbers
/// are comparable -- take it from `local-pca`'s dip. Returns the curve and the
/// null `d^2/D`.
pub fn tangent_overlap(
    c: &Cloud,
    nbr: &[Vec<(f64, usize)>],
    d: usize,
) -> (Vec<OverlapPoint>, f64) {
    let (m, dim) = (c.len(), c.coords.ncols());
    let d = d.clamp(1, dim.saturating_sub(1));
    if m < 32 || d == 0 || nbr.len() != m {
        return (Vec::new(), 0.0);
    }
    // Two different neighbourhoods, deliberately. Hops are counted on `nbr`,
    // the shared k-NN graph every other row uses, so "graph distance" means the
    // same thing here as it does in `fiedler` and `ricci`. The tangent *bases*
    // need their own, much larger: a d-dimensional subspace fitted from k+1
    // points is noise unless k is comfortably bigger than d, and the shared
    // graph is only 15-NN. Passing that in fitted 21-dimensional subspaces
    // through 16 points, and silently returned nothing whenever d >= 15.
    let k = (8 * d).clamp(64, 256).min(m - 2);
    if k <= d {
        return (Vec::new(), 0.0);
    }
    let wide = c.knn(k);
    if wide.is_empty() {
        return (Vec::new(), 0.0);
    }

    let stride = m.div_ceil(MAX_CENTRES).max(1);
    let centres: Vec<usize> = (0..m).step_by(stride).collect();
    let bases: Vec<Mat<f64>> = centres
        .par_iter()
        .map(|&i| tangent_basis(c, i, &wide[i][..k], d))
        .collect();

    let hops = hop_distances_from(&centres, nbr, m);
    let null = (d * d) as f64 / dim as f64;

    let mut sum = vec![0.0f64; MAX_HOPS + 1];
    let mut cnt = vec![0usize; MAX_HOPS + 1];
    for a in 0..centres.len() {
        for b in (a + 1)..centres.len() {
            let h = hops[a][centres[b]];
            if h == usize::MAX {
                continue; // different components: no distance to bin it at
            }
            let bin = h.min(MAX_HOPS);
            sum[bin] += frobenius_overlap(&bases[a], &bases[b]);
            cnt[bin] += 1;
        }
    }

    let curve = (1..=MAX_HOPS)
        .filter(|&h| cnt[h] > 0)
        .map(|h| OverlapPoint {
            hops: h,
            pairs: cnt[h],
            shared: sum[h] / cnt[h] as f64,
            excess: sum[h] / cnt[h] as f64 - null,
        })
        .collect();
    (curve, null)
}

/// Orthonormal basis for the `d` leading directions of the neighbourhood, in
/// the embedding's own frame.
///
/// The covariance is `dim x dim` and dim is at most `SCORE_K`, so this is a
/// small dense EVD -- no need for an SVD of the point matrix.
fn tangent_basis(c: &Cloud, centre: usize, nbr: &[(f64, usize)], d: usize) -> Mat<f64> {
    let dim = c.coords.ncols();
    let pts: Vec<usize> = std::iter::once(centre).chain(nbr.iter().map(|&(_, j)| j)).collect();
    let n = pts.len() as f64;
    let mean: Vec<f64> = (0..dim)
        .map(|j| pts.iter().map(|&i| c.coords.read(i, j)).sum::<f64>() / n)
        .collect();
    let cov = Mat::from_fn(dim, dim, |a, b| {
        pts.iter()
            .map(|&i| (c.coords.read(i, a) - mean[a]) * (c.coords.read(i, b) - mean[b]))
            .sum::<f64>()
            / n
    });
    let evd = SelfAdjointEigendecomposition::new(cov.as_ref(), Side::Lower);
    let u = evd.u();
    // faer returns ascending, so the leading directions are the last columns.
    Mat::from_fn(dim, d, |i, t| u.read(i, dim - 1 - t))
}

/// `||A^T B||_F^2` = sum of squared cosines of the principal angles.
fn frobenius_overlap(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let (dim, da, db) = (a.nrows(), a.ncols(), b.ncols());
    (0..da)
        .map(|i| {
            (0..db)
                .map(|j| {
                    let dot: f64 = (0..dim).map(|r| a.read(r, i) * b.read(r, j)).sum();
                    dot * dot
                })
                .sum::<f64>()
        })
        .sum()
}

/// Hop distance from each centre to every node, by BFS.
///
/// From the centres only, not all-pairs: 256 traversals rather than m, which is
/// the difference between free and the dominant cost at an 8000-cell cloud.
fn hop_distances_from(
    centres: &[usize],
    nbr: &[Vec<(f64, usize)>],
    m: usize,
) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (i, n) in nbr.iter().enumerate() {
        for &(_, j) in n {
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    centres
        .par_iter()
        .map(|&s| {
            let mut dist = vec![usize::MAX; m];
            dist[s] = 0;
            let mut q = std::collections::VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }
            dist
        })
        .collect()
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

    /// The overlap has to be the shared dimension count, exactly, on subspaces
    /// built by hand: a basis with itself shares all of its dimensions, two
    /// orthogonal ones share none, and a partial overlap counts the plane.
    #[test]
    fn overlap_counts_shared_dimensions() {
        let e = |cols: &[usize]| Mat::from_fn(6, cols.len(), |i, t| (i == cols[t]) as u8 as f64);
        let a = e(&[0, 1, 2]);
        assert!((frobenius_overlap(&a, &a) - 3.0).abs() < 1e-12);
        assert!(frobenius_overlap(&a, &e(&[3, 4, 5])).abs() < 1e-12);
        assert!((frobenius_overlap(&a, &e(&[1, 2, 3])) - 2.0).abs() < 1e-12);
        // Rotating within a subspace must not change what it shares.
        let s = 0.5f64.sqrt();
        let rot = Mat::from_fn(6, 3, |i, t| match (i, t) {
            (0, 0) | (1, 1) => s,
            (1, 0) => s,
            (0, 1) => -s,
            (2, 2) => 1.0,
            _ => 0.0,
        });
        assert!((frobenius_overlap(&a, &rot) - 3.0).abs() < 1e-10);
    }

    /// The claim the row exists to make. A flat patch has one tangent space
    /// everywhere, so overlap must stay near d at every distance. A curved
    /// object -- here a 2-sphere, whose tangent plane rotates all the way to
    /// orthogonal at the antipode -- must decay.
    #[test]
    fn flat_holds_its_tangent_space_and_curved_does_not() {
        let (n, dim) = (1500, 8);
        let mut r = lcg(2024);

        // A 3-flat in R^8: same tangent space at every point.
        let basis: Vec<f64> = (0..3 * dim).map(|_| r() - 0.5).collect();
        let lat: Vec<f64> = (0..n * 3).map(|_| r()).collect();
        let flat = Mat::from_fn(n, dim, |i, j| {
            (0..3).map(|k| lat[i * 3 + k] * basis[k * dim + j]).sum()
        });
        let c = Cloud::new(&flat, n);
        let nbr = c.knn(80);
        let (curve, null) = tangent_overlap(&c, &nbr, 3);
        assert!(curve.len() >= 3, "only {} bins", curve.len());
        let far = curve.last().unwrap();
        assert!(
            far.shared > 2.5,
            "flat patch lost its tangent space: {:.2} at {} hops (null {null:.2})",
            far.shared,
            far.hops
        );

        // A circle in R^8: the tangent *line* rotates with position, which is
        // the cleanest case where the same manifold offers different directions
        // in different places. A sphere is the obvious choice and a bad one --
        // its poles collapse under uniform parameter sampling, and a
        // neighbourhood of k points covers a cap wide enough that the plane
        // barely turns across the hop range.
        let circle = Mat::from_fn(2000, dim, |i, j| {
            let t = i as f64 / 2000.0 * std::f64::consts::TAU;
            match j {
                0 => t.cos(),
                1 => t.sin(),
                _ => 0.0,
            }
        });
        let c = Cloud::new(&circle, 2000);
        let nbr = c.knn(64);
        let (curve, _) = tangent_overlap(&c, &nbr, 1);
        let (near, far) = (&curve[0], curve.last().unwrap());
        assert!(
            near.shared > 0.9,
            "adjacent tangents should be near-parallel, got {:.2}",
            near.shared
        );
        assert!(
            far.shared < near.shared - 0.25,
            "circle did not rotate its tangent: {:.2} at {} hops -> {:.2} at {} hops",
            near.shared,
            near.hops,
            far.shared,
            far.hops
        );
    }
}
