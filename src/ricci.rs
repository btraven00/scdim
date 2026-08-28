//! Ollivier-Ricci curvature of the k-NN graph.
//!
//! For an edge (x, y), put a lazy random-walk measure on each endpoint --
//! mass `alpha` on the point itself, `(1-alpha)/k` on each of its k neighbours
//! -- and compare the cost of moving one to the other against the distance
//! between them:
//!
//!   kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y)
//!
//! Negative curvature means the two neighbourhoods are *harder* to align than
//! the endpoints are far apart: the edge is a bridge, with little overlap
//! between the neighbourhoods it joins. Positive curvature means the
//! neighbourhoods overlap heavily, as inside a cluster. So the negatively
//! curved cells are the ones sitting on bottlenecks -- branch points of a
//! trajectory, or the thin joins between cell types that keep `betti0` and
//! `fiedler` from calling the graph partitioned.
//!
//! This is the local version of those two global verdicts. They answer "is this
//! one piece?"; this answers "which cells are the seams?".
//!
//! ## Exact, not entropic
//!
//! W1 is solved exactly, by min-cost flow. The usual shortcut is Sinkhorn with
//! an entropic penalty, but that biases W1 *upward* and therefore kappa
//! downward -- straight into the quantity being counted here, which is the
//! number of cells with kappa < 0. Each transport problem is only (k+1)x(k+1),
//! so exactness costs little: ~30k edges at k=15 is around a second.
//!
//! ## Ground metric
//!
//! Euclidean distance in the embedding, not graph shortest-path distance.
//! Shortest paths would need an all-pairs solve on the k-NN graph per edge;
//! the cells already live in a metric space, so the ambient distance is both
//! cheaper and better behaved than a hop count.

use faer::Mat;
use rayon::prelude::*;

use crate::rank::Estimate;
use crate::twonn::strided_rows;

/// Per-cell curvature: the mean Ollivier-Ricci curvature of the edges at each
/// node. `alpha` is the laziness of the random walk (0.5 is the usual choice).
pub fn node_curvature(x: &Mat<f64>, max_points: usize, k: usize, alpha: f64) -> Vec<f64> {
    let rows = strided_rows(x.nrows(), max_points);
    let m = rows.len();
    if m < k + 2 {
        return Vec::new();
    }
    let sub = Mat::from_fn(m, x.ncols(), |i, j| x.read(rows[i], j));
    let gram = sub.as_ref() * sub.as_ref().transpose();
    let diag: Vec<f64> = (0..m).map(|i| gram.read(i, i)).collect();
    let dist = |a: usize, b: usize| (diag[a] + diag[b] - 2.0 * gram.read(a, b)).max(0.0).sqrt();

    let mut nbr: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(m);
    for i in 0..m {
        buf.clear();
        buf.extend((0..m).filter(|&j| j != i).map(|j| (dist(i, j), j)));
        buf.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        nbr.push(buf[..k].iter().map(|&(_, j)| j).collect());
    }

    // Edge set of the symmetrised graph, each edge once.
    let edges: Vec<(usize, usize)> = (0..m)
        .flat_map(|i| nbr[i].iter().map(move |&j| (i.min(j), i.max(j))))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let kappas: Vec<f64> = edges
        .par_iter()
        .map(|&(a, b)| {
            let d = dist(a, b);
            if d <= 0.0 {
                return 0.0;
            }
            let (sa, pa) = measure(a, &nbr[a], alpha);
            let (sb, pb) = measure(b, &nbr[b], alpha);
            let cost: Vec<Vec<f64>> = pa
                .iter()
                .map(|&u| pb.iter().map(|&v| dist(u, v)).collect())
                .collect();
            1.0 - wasserstein1(&sa, &sb, &cost) / d
        })
        .collect();

    let mut sum = vec![0.0f64; m];
    let mut cnt = vec![0u32; m];
    for (&(a, b), &kp) in edges.iter().zip(&kappas) {
        sum[a] += kp;
        sum[b] += kp;
        cnt[a] += 1;
        cnt[b] += 1;
    }
    (0..m)
        .map(|i| if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { 0.0 })
        .collect()
}

/// Lazy random-walk measure at `i`: `(masses, support)`.
fn measure(i: usize, nbr: &[usize], alpha: f64) -> (Vec<f64>, Vec<usize>) {
    let mut mass = Vec::with_capacity(nbr.len() + 1);
    let mut sup = Vec::with_capacity(nbr.len() + 1);
    if alpha > 0.0 {
        mass.push(alpha);
        sup.push(i);
    }
    let w = (1.0 - alpha) / nbr.len() as f64;
    for &j in nbr {
        mass.push(w);
        sup.push(j);
    }
    (mass, sup)
}

/// Exact 1-Wasserstein distance between two discrete measures of equal total
/// mass, by min-cost flow: successive shortest paths on *reduced* costs.
///
/// Kept general over unequal supports and unequal per-atom masses, which the
/// lazy walk needs -- the centre carries `alpha` and each neighbour
/// `(1-alpha)/k`, so this is a transportation problem, not an assignment.
///
/// Potentials (Johnson) keep every reduced cost non-negative so the inner
/// search can be Dijkstra. That is not a micro-optimisation over Bellman-Ford:
/// with raw costs the residual graph acquires zero-cost cycles as soon as two
/// neighbourhoods share a point, and a label-correcting search relaxes them
/// forever once round-off exceeds the improvement threshold. Distances here are
/// O(10-100), where f64 spacing is ~1e-14, so an absolute epsilon cannot
/// separate "improved" from "rounded" -- Dijkstra on non-negative reduced costs
/// removes the failure mode instead of tuning around it.
pub fn wasserstein1(supply: &[f64], demand: &[f64], cost: &[Vec<f64>]) -> f64 {
    const EPS: f64 = 1e-12;
    let (ns, nd) = (supply.len(), demand.len());
    let (src, snk) = (ns + nd, ns + nd + 1);
    let n = ns + nd + 2;

    // Arcs as flat (to, cap, cost, rev) lists per node.
    let mut g: Vec<Vec<(usize, f64, f64, usize)>> = vec![Vec::new(); n];
    let add = |g: &mut Vec<Vec<(usize, f64, f64, usize)>>, u: usize, v: usize, c: f64, w: f64| {
        let (iu, iv) = (g[u].len(), g[v].len());
        g[u].push((v, c, w, iv));
        g[v].push((u, 0.0, -w, iu));
    };
    let total_mass: f64 = supply.iter().sum();
    for i in 0..ns {
        add(&mut g, src, i, supply[i], 0.0);
        for j in 0..nd {
            add(&mut g, i, ns + j, total_mass, cost[i][j]);
        }
    }
    for j in 0..nd {
        add(&mut g, ns + j, snk, demand[j], 0.0);
    }

    let mut pot = vec![0.0f64; n];
    let mut total = 0.0f64;
    let mut shipped = 0.0f64;
    while shipped < total_mass - EPS {
        // Dijkstra over reduced costs w + pot[u] - pot[v], all >= 0.
        let mut d = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut seen = vec![false; n];
        d[src] = 0.0;
        loop {
            let u = (0..n)
                .filter(|&v| !seen[v] && d[v].is_finite())
                .min_by(|&a, &b| d[a].total_cmp(&d[b]));
            let Some(u) = u else { break };
            seen[u] = true;
            for (idx, &(v, cap, w, _)) in g[u].iter().enumerate() {
                if cap <= EPS || seen[v] {
                    continue;
                }
                let rw = w + pot[u] - pot[v];
                if d[u] + rw.max(0.0) < d[v] {
                    d[v] = d[u] + rw.max(0.0);
                    prev[v] = Some((u, idx));
                }
            }
        }
        if !d[snk].is_finite() {
            break; // no augmenting path: total mass is unroutable
        }
        for v in 0..n {
            if d[v].is_finite() {
                pot[v] += d[v];
            }
        }
        // Bottleneck, then push.
        let (mut f, mut v) = (f64::INFINITY, snk);
        while let Some((u, idx)) = prev[v] {
            f = f.min(g[u][idx].1);
            v = u;
        }
        if !(f > EPS) {
            break;
        }
        let mut v = snk;
        while let Some((u, idx)) = prev[v] {
            g[u][idx].1 -= f;
            let rev = g[u][idx].3;
            g[v][rev].1 += f;
            total += f * g[u][idx].2;
            v = u;
        }
        shipped += f;
    }
    total
}

/// How many cells sit on a bottleneck.
pub fn curvature_summary(kappa: &[f64]) -> Estimate {
    if kappa.is_empty() {
        return Estimate {
            name: "ricci-neg",
            rank: 0,
            detail: "too few points for a k-NN graph".to_string(),
            pvalues: Vec::new(),
        };
    }
    let n = kappa.len();
    let neg = kappa.iter().filter(|&&k| k < 0.0).count();
    let mean = kappa.iter().sum::<f64>() / n as f64;
    let mut sorted = kappa.to_vec();
    sorted.sort_by(f64::total_cmp);
    Estimate {
        name: "ricci-neg",
        rank: neg,
        detail: format!(
            "{neg}/{n} cells ({:.1}%) negatively curved; mean kappa {mean:+.3}, \
             range {:+.3} to {:+.3}, median {:+.3}",
            100.0 * neg as f64 / n as f64,
            sorted[0],
            sorted[n - 1],
            sorted[n / 2]
        ),
        pvalues: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transport solver, against costs that can be worked out by hand.
    #[test]
    fn wasserstein_matches_hand_calculation() {
        // One unit of mass moved a distance of 3.
        assert!((wasserstein1(&[1.0], &[1.0], &[vec![3.0]]) - 3.0).abs() < 1e-9);

        // Points at 0 and 1 -> points at 2 and 3. Optimal is 0->2, 1->3, at a
        // cost of 2; the crossing plan costs 3+1 = 4, so a greedy or misrouted
        // solver gives a different answer here.
        let c = vec![vec![2.0, 3.0], vec![1.0, 2.0]];
        assert!((wasserstein1(&[0.5, 0.5], &[0.5, 0.5], &c) - 2.0).abs() < 1e-9);

        // Unequal atom masses: the transportation case the lazy walk needs.
        // 0.75 at A and 0.25 at B; sinks want 0.5 each. Costs make A->0 free,
        // so 0.5 goes A->0, 0.25 goes A->1 at 1.0, 0.25 goes B->1 at 2.0.
        let c = vec![vec![0.0, 1.0], vec![5.0, 2.0]];
        let got = wasserstein1(&[0.75, 0.25], &[0.5, 0.5], &c);
        assert!((got - 0.75).abs() < 1e-9, "got {got}");
    }

    /// Regression: the shape and magnitude the CLI actually produces. Costs of
    /// order 50 with shared zero-cost support is what made a label-correcting
    /// search loop forever -- f64 spacing at that magnitude swamps any absolute
    /// improvement threshold.
    #[test]
    fn w1_terminates_at_cli_support_size() {
        // support 16 each (k=15 + lazy centre), unequal masses, many zero costs
        // from overlapping neighbourhoods -- the CLI's actual shape.
        let k = 15usize;
        let a = 0.5;
        let mut sup = vec![a];
        sup.extend(std::iter::repeat((1.0 - a) / k as f64).take(k));
        let mut s = 99u64;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 11) as f64 / (1u64<<53) as f64 };
        for scale in [1.0, 50.0, 1e4] {
            let cost: Vec<Vec<f64>> = (0..k + 1)
                .map(|i| (0..k + 1).map(|j| if i == j { 0.0 } else { scale * r() }).collect())
                .collect();
            let w = wasserstein1(&sup, &sup, &cost);
            assert!(w.is_finite() && w >= 0.0, "scale {scale}: w = {w}");
        }
    }

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A dense isotropic blob is clique-like everywhere: neighbourhoods overlap
    /// heavily, so curvature is positive and almost nothing is a bottleneck.
    #[test]
    fn blob_is_positively_curved() {
        let mut r = lcg(4);
        let x = Mat::from_fn(400, 4, |_, _| r() + r() + r());
        let k = node_curvature(&x, 400, 12, 0.5);
        let neg = k.iter().filter(|&&v| v < 0.0).count();
        assert!(neg * 10 < k.len(), "{neg}/{} negatively curved", k.len());
    }

    /// The case the diagnostic exists for: two blobs joined by a thin bridge.
    /// The bridge cells must be the negatively curved ones, and they must be
    /// more negative than the blob interiors.
    #[test]
    fn bridge_cells_are_negatively_curved() {
        let mut r = lcg(9);
        let (blob, span) = (250usize, 40usize);
        let n = 2 * blob + span;
        let x = Mat::from_fn(n, 3, |i, j| {
            if i < 2 * blob {
                // two tight blobs, centred at x = 0 and x = 30
                let c = if i < blob { 0.0 } else { 30.0 };
                if j == 0 { c + r() } else { r() }
            } else {
                // a thin filament joining them along x
                let t = (i - 2 * blob) as f64 / span as f64;
                if j == 0 { 1.0 + 28.0 * t } else { 0.5 + 0.02 * r() }
            }
        });
        let k = node_curvature(&x, n, 10, 0.5);
        let bridge: f64 = k[2 * blob..].iter().sum::<f64>() / span as f64;
        let inside: f64 = k[..blob].iter().sum::<f64>() / blob as f64;
        assert!(bridge < 0.0, "bridge mean kappa {bridge:+.3} not negative");
        assert!(
            bridge < inside - 0.1,
            "bridge {bridge:+.3} not clearly below blob interior {inside:+.3}"
        );
    }
}
