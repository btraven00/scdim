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
//! ## The curve is a within-run shape; the depth rows are cross-run numbers
//!
//! Measured across clouds of 2000/4000/8000 cells, this row splits cleanly.
//!
//! **Stable, and comparable between runs:** the correlation of the leading
//! shared direction with log depth (0-3.5% drift on four datasets), *which* PC
//! holds the depth axis (identical at every size, including the one dataset
//! where it is not PC1), the fraction of that PC it occupies (0-4%), and the
//! null verdict.
//!
//! **Not comparable between runs:** the overlap curve itself. It falls 8-36%
//! across a 4x cloud on every dataset. Calibrating the x-axis in distance
//! rather than hops does not fix it -- at a fixed radius the drift is 34%,
//! worse than at a fixed hop -- because the cause is the basis neighbourhood
//! shrinking physically as the cloud densifies, not the hop axis. Read the
//! curve's shape within one run, and do not compare its values across
//! `--geom-cells` settings.
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

/// Draws of the whole ensemble used to build the `lambda_1` null. The
/// statistic is an extremum, so what matters is the max over draws; 20 gives
/// a one-in-twenty-one exceedance to quote and costs well under a second.
pub const NULL_DRAWS: usize = 20;

/// Above this fraction of `d`, the random-subspace null leaves too little room
/// for the measurement to say much.
pub const NULL_WARN: f64 = 0.3;

/// Everything the tangent pass produces.
pub struct Tangent {
    /// Overlap against graph distance.
    pub curve: Vec<OverlapPoint>,
    /// `d^2/D`: shared dimensions between two *random* subspaces. The baseline
    /// for `OverlapPoint::excess`.
    pub null: f64,
    /// Tangent dimension used.
    pub d: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Eigenvalues of the mean projector `M = (1/N) sum P_i`, descending.
    ///
    /// Eigenvalue j is the fraction of regions whose tangent space contains
    /// direction j: 1 means every region varies along it, 0 means none does.
    /// `tr(M) = d` always, so a completely unstructured cloud spreads them flat
    /// at `d/D` -- **that is the chance level for this spectrum, not `d^2/D`**,
    /// which is the chance level for the pairwise overlap. The two are easy to
    /// confuse and differ by a factor of d.
    ///
    /// This is Flury's Common Principal Components (1984) applied to local
    /// tangent spaces: the leading eigenvectors span the subspace every region
    /// shares, and the geometry needs no clustering to find it.
    ///
    pub spectrum: Vec<f64>,
    /// Per-rank maximum of the null spectrum over [`NULL_DRAWS`] draws.
    ///
    /// Counting observed eigenvalues that beat their own rank's null is the
    /// dimension of the shared subspace. Comparing every observed eigenvalue to
    /// `lambda_1`'s null instead would be the wrong test twice over -- the null
    /// spectrum is itself a decaying curve, and an order statistic has to be
    /// compared with the same order statistic.
    pub null_spectrum: Vec<f64>,
    /// Eigenvectors of the mean projector, `D x D`, columns ordered to match
    /// [`Tangent::spectrum`].
    ///
    /// These *are* the shared directions. Kept so that something can be asked
    /// of them -- most usefully whether they are technical, since library size,
    /// ambient RNA and cell cycle all vary inside every region and are
    /// therefore exactly what a shared-direction detector will find first.
    ///
    /// All of them, not a leading slice: D is at most `SCORE_K`, so the whole
    /// matrix is under 80 KB, and the shared subspace has come back as wide as
    /// 31 directions -- a cap would have silently truncated the projector whose
    /// diagonal says which PCs are contaminated.
    pub vectors: Mat<f64>,
    /// `lambda_1` of the mean projector under the null, as `(mean, max)` over
    /// [`NULL_DRAWS`] draws.
    ///
    /// `d/D` is where the null puts the *mean* of the spectrum, not its top: a
    /// finite number of random frames fluctuates, and the largest of D
    /// eigenvalues sits well above d/D by construction. Comparing an observed
    /// `lambda_1` to `d/D` therefore overstates it, sometimes badly -- at
    /// d/D = 0.48 a null `lambda_1` near 0.9 is free.
    ///
    /// The null depends only on the number of frames, d and D, never on the
    /// data, so it costs one small computation rather than a resampling of the
    /// cloud. Deterministically seeded: a diagnostic you cannot rerun is not a
    /// diagnostic.
    pub null_lambda1: (f64, f64),
}

impl Tangent {
    /// Fraction of each PC's direction that lies inside the shared subspace:
    /// `diag(sum_c v_c v_c^T)`, read in the PC basis.
    ///
    /// In [0, 1] per PC, summing to `k` across them. This is the actionable
    /// output -- "PC1 is 61% shared subspace, PC4 is 3%" says which components
    /// to distrust, which a global correlation cannot.
    pub fn contamination(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.vectors.ncols());
        (0..self.dim)
            .map(|j| (0..k).map(|c| self.vectors.read(j, c).powi(2)).sum())
            .collect()
    }

    /// Dimension of the shared subspace: eigenvalues beating their own rank's
    /// null on every draw.
    pub fn shared_dims(&self) -> usize {
        self.spectrum
            .iter()
            .zip(&self.null_spectrum)
            .take_while(|(obs, null)| obs > null)
            .count()
    }
}

impl Tangent {
    /// Chance level for one eigenvalue of the mean projector.
    pub fn chance(&self) -> f64 {
        self.d as f64 / self.dim as f64
    }

    /// How many directions the shared structure spreads over:
    /// `(sum lambda)^2 / sum lambda^2`, the same participation ratio
    /// `local-pca` uses, and bounded by `[d, D]`.
    ///
    /// At the lower bound every region shares the *same* d directions -- one
    /// common tangent space. At the upper bound the regions between them cover
    /// the embedding evenly and share nothing. It needs no threshold, which is
    /// why it is the headline rather than a count of eigenvalues above a cut.
    pub fn concentration(&self) -> f64 {
        let sq: f64 = self.spectrum.iter().map(|l| l * l).sum();
        if sq > 0.0 {
            (self.d as f64).powi(2) / sq
        } else {
            f64::NAN
        }
    }
}

/// One distance bin of the overlap curve.
pub struct OverlapPoint {
    /// Graph distance in k-NN hops. `hops == MAX_HOPS` means "at least this".
    ///
    /// Not comparable across cloud sizes: a denser graph reaches less far per
    /// hop, so the same bin is a shorter distance. Measured across clouds of
    /// 2000/4000/8000 the overlap at a fixed hop count fell by 8-36% on every
    /// dataset for that reason alone. Read `mean_r` instead when comparing runs.
    pub hops: usize,
    /// Mean Euclidean distance between the pairs in this bin -- the hop axis in
    /// the data's own units, and the one that is comparable between runs.
    pub mean_r: f64,
    pub pairs: usize,
    /// Mean `tr(P_x P_y)`: shared dimensions, including the chance baseline.
    pub shared: f64,
    /// The same, less the random-subspace null.
    pub excess: f64,
    /// Standard deviation of `shared` within this bin, over the same pairs.
    ///
    /// The mean alone cannot tell a smoothly rotating manifold from two rigid
    /// pieces meeting at an angle: both give a decaying average. A smooth
    /// manifold makes overlap a function of distance, so the spread at a fixed
    /// distance stays small; distinct pieces put within-piece and across-piece
    /// pairs in the same bin with very different overlaps, and the spread
    /// blows up.
    pub sd: f64,
    /// `excess` as a fraction of the room there is to measure in.
    ///
    /// The null is a ceiling as well as a floor: the largest excess possible is
    /// `d - d^2/D`, which at d = 14 in D = 29 is 7.2 rather than 14. Raw excess
    /// is therefore not comparable between datasets with different d and D --
    /// this is. 1.0 means two identical tangent spaces, 0.0 means no more
    /// overlap than two random subspaces.
    pub frac: f64,
}

/// Tangent-space overlap against graph distance.
///
/// `d` is the tangent dimension to use, the same for every point so the numbers
/// are comparable -- take it from `local-pca`'s dip. Returns the curve and the
/// null `d^2/D`.
pub fn tangent_overlap(c: &Cloud, nbr: &[Vec<(f64, usize)>], d: usize) -> Tangent {
    let (m, dim) = (c.len(), c.coords.ncols());
    let d = d.clamp(1, dim.saturating_sub(1));
    let empty = || Tangent {
        curve: Vec::new(),
        null: 0.0,
        d: 0,
        dim,
        spectrum: Vec::new(),
        vectors: Mat::zeros(0, 0),
        null_spectrum: Vec::new(),
        null_lambda1: (0.0, 0.0),
    };
    if m < 32 || d == 0 || nbr.len() != m {
        return empty();
    }
    // Two different neighbourhoods, deliberately. Hops are counted on `nbr`,
    // the shared k-NN graph every other row uses, so "graph distance" means the
    // same thing here as it does in `fiedler` and `ricci`. The tangent *bases*
    // need their own, much larger: a d-dimensional subspace fitted from k+1
    // points is noise unless k is comfortably bigger than d, and the shared
    // graph is only 15-NN. Passing that in fitted 21-dimensional subspaces
    // through 16 points, and silently returned nothing whenever d >= 15.
    // A fixed count, which is what k-NN local PCA normally does. A fixed
    // *fraction* of the cloud was tried -- the correction that took
    // `local-pca`'s drift from 24-46% to 3-6% -- and it does not transfer:
    // fixed-count overlap drifts down 32% across a 4x cloud, fixed-fraction
    // drifts *up* 27%, and the truth is somewhere between with no simple rule
    // reaching it. Overlap depends on the neighbourhood radius relative to the
    // manifold's curvature scale, which neither a count nor a fraction pins.
    // The simpler of two equally-wrong options, with the limitation documented.
    let k = (8 * d).clamp(64, 256).min(m - 2);
    if k <= d {
        return empty();
    }
    let wide = c.knn(k);
    if wide.is_empty() {
        return empty();
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
    let mut sq = vec![0.0f64; MAX_HOPS + 1];
    let mut dsum = vec![0.0f64; MAX_HOPS + 1];
    let mut cnt = vec![0usize; MAX_HOPS + 1];
    for a in 0..centres.len() {
        for b in (a + 1)..centres.len() {
            let h = hops[a][centres[b]];
            if h == usize::MAX {
                continue; // different components: no distance to bin it at
            }
            let bin = h.min(MAX_HOPS);
            let ov = frobenius_overlap(&bases[a], &bases[b]);
            sum[bin] += ov;
            sq[bin] += ov * ov;
            dsum[bin] += c.dist(centres[a], centres[b]);
            cnt[bin] += 1;
        }
    }

    // d < dim is enforced above, so the room is strictly positive.
    let room = d as f64 - null;
    let curve = (1..=MAX_HOPS)
        .filter(|&h| cnt[h] > 0)
        .map(|h| {
            let n = cnt[h] as f64;
            let shared = sum[h] / n;
            OverlapPoint {
                sd: (sq[h] / n - shared * shared).max(0.0).sqrt(),
                hops: h,
                mean_r: dsum[h] / cnt[h] as f64,
                pairs: cnt[h],
                shared,
                excess: shared - null,
                frac: (shared - null) / room,
            }
        })
        .collect();

    // M = mean of the projectors. tr(M) = d by construction, so its spectrum
    // says how those d dimensions are distributed over the embedding rather
    // than how many there are.
    let n = bases.len() as f64;
    let mut mp = Mat::<f64>::zeros(dim, dim);
    for u in &bases {
        for a in 0..dim {
            for b in 0..=a {
                let v: f64 = (0..d).map(|t| u.read(a, t) * u.read(b, t)).sum();
                mp.write(a, b, mp.read(a, b) + v / n);
                if a != b {
                    mp.write(b, a, mp.read(a, b));
                }
            }
        }
    }
    let evd = SelfAdjointEigendecomposition::new(mp.as_ref(), Side::Lower);
    let mut spectrum: Vec<f64> = (0..dim).map(|i| evd.s().column_vector().read(i)).collect();
    spectrum.reverse();
    let ev = evd.u();
    // faer returns ascending, so the leading directions are the last columns.
    let vectors = Mat::from_fn(dim, dim, |i, t| ev.read(i, dim - 1 - t));

    let null_spectrum = null_spectrum(bases.len(), d, dim);
    let null_lambda1 = (
        // Mean of the top eigenvalue is not recoverable from the per-rank max,
        // so report the max twice rather than invent one.
        null_spectrum.first().copied().unwrap_or(0.0),
        null_spectrum.first().copied().unwrap_or(0.0),
    );
    Tangent { curve, null, d, dim, spectrum, vectors, null_spectrum, null_lambda1 }
}

/// The mean projector's spectrum when the tangent spaces carry no shared
/// structure at all: per rank, the largest value seen over [`NULL_DRAWS`]
/// ensembles of uniformly random orthonormal d-frames.
///
/// This is the whole test. An observed eigenvalue means something only if it
/// clears what N random frames produce for free, and at a large d/D that is
/// most of the way to 1 -- at d/D = 0.48 the null's own top eigenvalue is 0.54.
pub fn null_spectrum(frames: usize, d: usize, dim: usize) -> Vec<f64> {
    if frames == 0 || d == 0 || d >= dim {
        return Vec::new();
    }
    let draws: Vec<Vec<f64>> = (0..NULL_DRAWS)
        .into_par_iter()
        .map(|draw| {
            let mut state = 0x5EED_u64 ^ (draw as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut unit = move || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
            };
            let mut m = Mat::<f64>::zeros(dim, dim);
            let w = 1.0 / frames as f64;
            for _ in 0..frames {
                let u = random_frame(dim, d, &mut unit);
                for a in 0..dim {
                    for b in 0..=a {
                        let v: f64 = (0..d).map(|t| u.read(a, t) * u.read(b, t)).sum();
                        m.write(a, b, m.read(a, b) + v * w);
                        if a != b {
                            m.write(b, a, m.read(a, b));
                        }
                    }
                }
            }
            let evd = SelfAdjointEigendecomposition::new(m.as_ref(), Side::Lower);
            let mut e: Vec<f64> = (0..dim).map(|i| evd.s().column_vector().read(i)).collect();
            e.reverse();
            e
        })
        .collect();
    // Per rank, the largest the null ever produced: an exceedance test at
    // 1/(NULL_DRAWS+1) for every eigenvalue, not just the first.
    (0..dim)
        .map(|j| draws.iter().map(|e| e[j]).fold(f64::MIN, f64::max))
        .collect()
}

/// A uniformly random orthonormal d-frame in R^dim: Gaussian columns, then
/// modified Gram-Schmidt. At these sizes (d <= 30, dim <= 100) that is cheaper
/// than reaching for a QR, and stable enough -- the columns start orthogonal in
/// expectation, so nothing is being subtracted away.
fn random_frame(dim: usize, d: usize, unit: &mut impl FnMut() -> f64) -> Mat<f64> {
    let mut gauss = || {
        // Box-Muller; the u1 floor keeps ln() finite.
        let (u1, u2) = (unit().max(1e-12), unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut u = Mat::<f64>::zeros(dim, d);
    for t in 0..d {
        for i in 0..dim {
            u.write(i, t, gauss());
        }
        for s in 0..t {
            let dot: f64 = (0..dim).map(|i| u.read(i, s) * u.read(i, t)).sum();
            for i in 0..dim {
                u.write(i, t, u.read(i, t) - dot * u.read(i, s));
            }
        }
        let nrm = (0..dim).map(|i| u.read(i, t).powi(2)).sum::<f64>().sqrt();
        for i in 0..dim {
            u.write(i, t, u.read(i, t) / nrm.max(1e-300));
        }
    }
    u
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

    /// The null has to behave like a null: a frame basis is orthonormal, and
    /// `lambda_1` sits above `d/D` (a finite ensemble fluctuates) but below 1,
    /// falling towards `d/D` as the ensemble grows. If it did not, comparing an
    /// observed `lambda_1` to `d/D` would be defensible -- it is not.
    #[test]
    fn null_lambda1_is_above_the_flat_level_and_shrinks() {
        let mut r = lcg(99);
        let f = random_frame(12, 4, &mut r);
        for a in 0..4 {
            for b in 0..4 {
                let dot: f64 = (0..12).map(|i| f.read(i, a) * f.read(i, b)).sum();
                let want = (a == b) as u8 as f64;
                assert!((dot - want).abs() < 1e-10, "frame not orthonormal at ({a},{b})");
            }
        }

        let flat = 4.0 / 12.0;
        let small = null_spectrum(32, 4, 12);
        let big = null_spectrum(512, 4, 12);
        assert!(small[0] > flat && small[0] < 1.0, "lambda_1 null {:.3}", small[0]);
        assert!(
            small.windows(2).all(|w| w[1] <= w[0] + 1e-9),
            "null spectrum not descending"
        );
        assert!(
            big[0] < small[0],
            "null did not shrink with ensemble size: {:.3} -> {:.3}",
            small[0],
            big[0]
        );
        assert!(big[0] > flat, "null fell below the flat level: {:.3}", big[0]);
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
        let t = tangent_overlap(&c, &nbr, 3);
        let (curve, null) = (&t.curve, t.null);
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
        let curve = &tangent_overlap(&c, &nbr, 1).curve;
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
