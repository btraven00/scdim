//! The shared point cloud: one Gram matrix and one k-NN graph, six heuristics.
//!
//! Every geometric diagnostic here wants the same two things: pairwise
//! distances among the same strided subsample, and (for two of them) the same
//! k-NN lists. Building them once is not just an optimisation of five identical
//! O(m^2) scans -- it is what lets the TwoNN scale analysis index its
//! decimation levels into an existing matrix instead of running a fresh BLAS
//! call per level per replicate.

use faer::Mat;
use rayon::prelude::*;

/// Distances among a strided subsample of the rows of an embedding.
///
/// Held as the Gram matrix rather than the distance matrix: the same memory,
/// and `d^2 = G_aa + G_bb - 2 G_ab` is cheaper than the BLAS-3 call that built
/// it. Indices are *local* -- 0..len(), not row numbers in the embedding.
pub struct Cloud {
    /// Rows of the embedding each local index came from.
    pub rows: Vec<usize>,
    /// The points themselves, `len() x embedding dim`.
    ///
    /// The Gram matrix is enough for every distance-based diagnostic, and was
    /// all this held for a while. Comparing two neighbourhoods' *tangent
    /// spaces* needs a common frame, though, which distances alone do not carry
    /// -- classical MDS reconstructs each neighbourhood in its own arbitrary
    /// basis. Keeping the coordinates costs 2000 x 65 x 8 = 1 MB.
    pub coords: Mat<f64>,
    gram: Mat<f64>,
    diag: Vec<f64>,
}

impl Cloud {
    /// A regular stride through the rows, capped at `max_points`.
    /// Deterministic, and cells are not stored in a meaningful order anyway.
    pub fn new(x: &Mat<f64>, max_points: usize) -> Self {
        let stride = x.nrows().div_ceil(max_points.max(1)).max(1);
        let rows: Vec<usize> = (0..x.nrows()).step_by(stride).collect();
        let m = rows.len();
        let sub = Mat::from_fn(m, x.ncols(), |i, j| x.read(rows[i], j));
        let gram = sub.as_ref() * sub.as_ref().transpose();
        let diag = (0..m).map(|i| gram.read(i, i)).collect();
        Cloud { rows, coords: sub, gram, diag }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Squared Euclidean distance. Clamped at zero: the Gram identity is
    /// exact in arithmetic and slightly negative in floating point.
    pub fn d2(&self, a: usize, b: usize) -> f64 {
        (self.diag[a] + self.diag[b] - 2.0 * self.gram.read(a, b)).max(0.0)
    }

    pub fn dist(&self, a: usize, b: usize) -> f64 {
        self.d2(a, b).sqrt()
    }

    /// The `k` nearest neighbours of every point as `(d^2, index)`, ascending.
    ///
    /// Empty when there are too few points to have k neighbours, which is how
    /// `fiedler` and `ricci` decline to report on a tiny input.
    ///
    /// Sorted, not merely selected: `fiedler` needs the k-th distance for its
    /// local scale, and sorting 15 elements per point is free next to the
    /// O(m^2) scan that found them.
    pub fn knn(&self, k: usize) -> Vec<Vec<(f64, usize)>> {
        let m = self.len();
        if k == 0 || m < k + 2 {
            return Vec::new();
        }
        (0..m)
            .into_par_iter()
            .map(|i| {
                let mut buf: Vec<(f64, usize)> = (0..m)
                    .filter(|&j| j != i)
                    .map(|j| (self.d2(i, j), j))
                    .collect();
                buf.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                buf.truncate(k);
                buf.sort_by(|a, b| a.0.total_cmp(&b.0));
                buf
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distances off the Gram must match the direct computation, and the k-NN
    /// lists must really be the k nearest, in order.
    #[test]
    fn distances_and_neighbours_are_right() {
        let mut s = 1u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = Mat::from_fn(60, 4, |_, _| r());
        let c = Cloud::new(&x, 60);
        assert_eq!(c.len(), 60);

        let direct = |a: usize, b: usize| {
            (0..4).map(|j| (x.read(a, j) - x.read(b, j)).powi(2)).sum::<f64>()
        };
        for (a, b) in [(0, 1), (7, 43), (59, 0), (5, 5)] {
            assert!((c.d2(a, b) - direct(a, b)).abs() < 1e-10);
        }

        let knn = c.knn(5);
        for (i, nn) in knn.iter().enumerate() {
            assert_eq!(nn.len(), 5);
            assert!(nn.windows(2).all(|w| w[1].0 >= w[0].0), "not sorted");
            let kth = nn[4].0;
            let closer = (0..60).filter(|&j| j != i && c.d2(i, j) < kth).count();
            assert!(closer <= 4, "point {i} has {closer} points inside its 5th neighbour");
        }

        // Too few points to have k neighbours: decline rather than panic.
        assert!(Cloud::new(&x, 3).knn(15).is_empty());
    }
}
