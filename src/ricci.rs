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

/// Ground metric for the transport problem. The choice is not cosmetic: it
/// changes the sign of the answer, so all three are exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Shortest path through the k-NN graph with Euclidean edge weights --
    /// the discrete geodesic. **The default, and what the literature uses.**
    ///
    /// It agrees with `Euclidean` for adjacent points (the direct edge is the
    /// shortest path) but grows correctly for points that are close in the
    /// ambient space yet far along the manifold, which is exactly the pair a
    /// curvature diagnostic has to price properly.
    Geodesic,
    /// Unweighted hop count. Every edge is d = 1, so W1 is measured in steps.
    /// Correct for an abstract graph, but it throws away the fact that some
    /// k-NN edges are far longer than others, and the degree heterogeneity of
    /// a symmetrised k-NN graph then drives kappa negative almost everywhere.
    Hops,
    /// Straight-line distance in the embedding.
    ///
    /// **Wrong for this purpose, and kept only to show why.** It lets mass
    /// travel through ambient space along chords the manifold does not
    /// contain, so the transport cost is systematically underestimated and
    /// kappa comes out positive almost everywhere -- on a branching
    /// hematopoiesis trajectory it reported 0/4874 negatively curved cells.
    Euclidean,
}

/// Per-cell curvature: the mean Ollivier-Ricci curvature of the edges at each
/// node. `alpha` is the laziness of the random walk (0.5 is the usual choice).
pub fn node_curvature(
    x: &Mat<f64>,
    max_points: usize,
    k: usize,
    alpha: f64,
    metric: Metric,
) -> Vec<f64> {
    let rows = strided_rows(x.nrows(), max_points);
    let m = rows.len();
    if m < k + 2 {
        return Vec::new();
    }
    let sub = Mat::from_fn(m, x.ncols(), |i, j| x.read(rows[i], j));
    let gram = sub.as_ref() * sub.as_ref().transpose();
    let diag: Vec<f64> = (0..m).map(|i| gram.read(i, i)).collect();
    let euclid = |a: usize, b: usize| (diag[a] + diag[b] - 2.0 * gram.read(a, b)).max(0.0).sqrt();

    let mut knn: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(m);
    for i in 0..m {
        buf.clear();
        buf.extend((0..m).filter(|&j| j != i).map(|j| (euclid(i, j), j)));
        buf.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        knn.push(buf[..k].iter().map(|&(_, j)| j).collect());
    }
    // The measure has to live on the *symmetrised* neighbourhood: k-NN is
    // directed, and a hub picked by many points but picking few has a much
    // larger true degree than k.
    let adj = symmetrise(&knn);

    let (edges, kappas) = match metric {
        Metric::Euclidean => curvature_edges(&adj, &euclid, alpha),
        Metric::Hops => {
            let hops = hop_distances(&adj);
            curvature_edges(&adj, &|a, b| hops[a][b] as f64, alpha)
        }
        Metric::Geodesic => {
            let geo = geodesic_distances(&adj, &euclid);
            curvature_edges(&adj, &|a, b| geo[a][b] as f64, alpha)
        }
    };

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

/// Undirected neighbour lists from directed k-NN lists.
pub fn symmetrise(knn: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut adj: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); knn.len()];
    for (i, nn) in knn.iter().enumerate() {
        for &j in nn {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }
    adj.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// All-pairs hop distance by BFS from every node.
///
/// Disconnected pairs get a large finite value rather than infinity: the
/// transport problem needs a finite cost matrix, and a component boundary is
/// exactly where curvature should be most negative, not undefined.
pub fn hop_distances(adj: &[Vec<usize>]) -> Vec<Vec<u16>> {
    let n = adj.len();
    let far = (n as u16).saturating_add(1);
    (0..n)
        .into_par_iter()
        .map(|s| {
            let mut d = vec![far; n];
            d[s] = 0;
            let mut q = std::collections::VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if d[v] == far {
                        d[v] = d[u] + 1;
                        q.push_back(v);
                    }
                }
            }
            d
        })
        .collect()
}

/// All-pairs shortest-path distance through `adj` with Euclidean edge weights:
/// the discrete geodesic on the k-NN graph.
///
/// Dijkstra from every node, in f32 -- at these sizes the matrix is the memory
/// cost (m^2 floats), and curvature does not need f64 in the ground metric.
/// Unreachable pairs get the largest finite distance seen, so a component
/// boundary prices as very expensive rather than undefined.
pub fn geodesic_distances(
    adj: &[Vec<usize>],
    w: &(dyn Fn(usize, usize) -> f64 + Sync),
) -> Vec<Vec<f32>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = adj.len();
    let mut out: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|s| {
            let mut d = vec![f32::INFINITY; n];
            d[s] = 0.0;
            let mut heap = BinaryHeap::new();
            heap.push((Reverse(ordered_float(0.0)), s));
            while let Some((Reverse(du), u)) = heap.pop() {
                let du = f32::from_bits(du ^ 0x8000_0000);
                if du > d[u] {
                    continue;
                }
                for &v in &adj[u] {
                    let nd = du + w(u, v) as f32;
                    if nd < d[v] {
                        d[v] = nd;
                        heap.push((Reverse(ordered_float(nd)), v));
                    }
                }
            }
            d
        })
        .collect();
    // Finite stand-in for disconnected pairs.
    let far = out
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
        * 4.0;
    for row in &mut out {
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = far.max(1.0);
            }
        }
    }
    out
}

/// Order-preserving bit pattern for non-negative f32, so it can go in a
/// `BinaryHeap` without pulling in an ordered-float dependency.
fn ordered_float(x: f32) -> u32 {
    x.to_bits() ^ 0x8000_0000
}

/// Ollivier-Ricci curvature of every undirected edge of `adj` under `dist`.
///
/// Split out from the point cloud so it can be checked against graphs whose
/// curvature is known in closed form -- see the tests.
pub fn curvature_edges(
    adj: &[Vec<usize>],
    dist: &(dyn Fn(usize, usize) -> f64 + Sync),
    alpha: f64,
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let edges: Vec<(usize, usize)> = (0..adj.len())
        .flat_map(|i| adj[i].iter().filter(move |&&j| j > i).map(move |&j| (i, j)))
        .collect();
    let kappas = edges
        .par_iter()
        .map(|&(a, b)| {
            let d = dist(a, b);
            if d <= 0.0 {
                return 0.0;
            }
            let (sa, pa) = measure(a, &adj[a], alpha);
            let (sb, pb) = measure(b, &adj[b], alpha);
            let cost: Vec<Vec<f64>> = pa
                .iter()
                .map(|&u| pb.iter().map(|&v| dist(u, v)).collect())
                .collect();
            1.0 - wasserstein1(&sa, &sb, &cost) / d
        })
        .collect();
    (edges, kappas)
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

/// How many cells sit on a bottleneck, and how deep the bottlenecks go.
///
/// # Why the sign test is not enough, and what replaces it
///
/// Counting `kappa < 0` is close to meaningless on its own. Flat space has zero
/// Ricci curvature, so in a featureless region the curvature distribution
/// straddles zero and roughly half the cells land negative on sampling noise
/// alone -- 39% negative is not evidence of anything. Nor does an absolute
/// threshold help: the spread of kappa depends on `k`, on the laziness `alpha`,
/// and on the local density, so a cut that isolates the tail on one dataset
/// sits in the bulk of the next.
///
/// `cut` is therefore in **robust z units** of the data's own curvature
/// distribution (Iglewicz-Hoaglin):
///
///   z_i = 0.6745 (kappa_i - median) / MAD
///
/// The 0.6745 makes MAD a consistent estimate of sigma for Gaussian data, so
/// `cut = 3.5` means the conventional outlier threshold whatever the scale.
/// Being a ratio of two quantities in the same units, it carries no units of
/// its own and transfers across datasets, k, and alpha.
///
/// The count alone would still be hard to read, so the summary also reports the
/// **tail asymmetry**: the number of cells below `-cut` against the number
/// above `+cut`. This needs no distributional assumption at all. Symmetric
/// noise around a flat mean gives asymmetry ~1; genuine bottlenecks put mass in
/// the left tail and nothing matching on the right, so asymmetry climbs. That
/// ratio, not the raw fraction, is the thing to read.
pub fn curvature_summary(kappa: &[f64], cut: f64) -> Estimate {
    if kappa.len() < 8 {
        return Estimate {
            name: "ricci-neg",
            rank: 0,
            detail: "too few points for a k-NN graph".to_string(),
            pvalues: Vec::new(),
        };
    }
    let n = kappa.len();
    let mut sorted = kappa.to_vec();
    sorted.sort_by(f64::total_cmp);
    let pct = |q: f64| sorted[((q * n as f64) as usize).min(n - 1)];
    let median = pct(0.5);

    let mut dev: Vec<f64> = kappa.iter().map(|k| (k - median).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let mad = dev[n / 2];
    if mad <= 0.0 {
        return Estimate {
            name: "ricci-neg",
            rank: 0,
            detail: format!("degenerate: every cell has kappa = {median:+.3}"),
            pvalues: Vec::new(),
        };
    }

    let z = |k: f64| 0.6745 * (k - median) / mad;
    let low = kappa.iter().filter(|&&k| z(k) < -cut).count();
    let high = kappa.iter().filter(|&&k| z(k) > cut).count();
    let neg = kappa.iter().filter(|&&k| k < 0.0).count();
    let asym = low as f64 / high.max(1) as f64;

    Estimate {
        name: "ricci-neg",
        rank: low,
        detail: format!(
            "{low}/{n} cells beyond -{cut} robust-z ({high} beyond +{cut}, tail asymmetry \
             {asym:.2}x); kappa median {median:+.3}, MAD {mad:.3}, min {:+.3}; \
             {:.0}% have kappa < 0",
            sorted[0],
            100.0 * neg as f64 / n as f64,
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

    /// Curvature of a graph whose value can be derived by hand, with hop
    /// distance and no laziness. Three cases, one of each sign.
    #[test]
    fn matches_closed_form_graph_curvature() {
        let kappa = |adj: &Vec<Vec<usize>>| {
            let hops = hop_distances(adj);
            let (e, k) = curvature_edges(adj, &|a, b| hops[a][b] as f64, 0.0);
            (e, k)
        };

        // K_5: m_x and m_y differ only by 1/(n-1) sitting on each other, one
        // hop apart, so W1 = 1/4 and kappa = 3/4 on every edge.
        let k5: Vec<Vec<usize>> = (0..5)
            .map(|i| (0..5).filter(|&j| j != i).collect())
            .collect();
        for v in kappa(&k5).1 {
            assert!((v - 0.75).abs() < 1e-9, "K_5 edge kappa {v}, expected 0.75");
        }

        // Cycle C_8: shift both neighbours one step, W1 = 1 = d, so kappa = 0.
        let c8: Vec<Vec<usize>> = (0..8usize).map(|i| vec![(i + 7) % 8, (i + 1) % 8]).collect();
        for v in kappa(&c8).1 {
            assert!(v.abs() < 1e-9, "C_8 edge kappa {v}, expected 0");
        }

        // Two 2-stars joined at their centres 0-1. The bridge must be
        // negative: one leaf pair is 3 hops apart, giving W1 = 5/3 against
        // d = 1, so kappa = -2/3.
        let stars: Vec<Vec<usize>> = vec![
            vec![1, 2, 3],
            vec![0, 4, 5],
            vec![0],
            vec![0],
            vec![1],
            vec![1],
        ];
        let (edges, ks) = kappa(&stars);
        let bridge = edges.iter().position(|&e| e == (0, 1)).unwrap();
        assert!(
            (ks[bridge] + 2.0 / 3.0).abs() < 1e-9,
            "bridge kappa {}, expected -2/3",
            ks[bridge]
        );
    }

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The scale-free cut has to be symmetric on flat space and left-heavy on
    /// a bottleneck. That asymmetry, not the raw fraction below zero, is what
    /// makes the count mean something.
    #[test]
    fn tail_asymmetry_separates_flat_from_bottleneck() {
        let asym = |e: &Estimate| {
            let s = e.detail.split("asymmetry ").nth(1).unwrap();
            s[..s.find('x').unwrap()].trim().parse::<f64>().unwrap()
        };

        let mut r = lcg(4);
        let flat = Mat::from_fn(600, 4, |_, _| r() + r() + r());
        let flat_k = node_curvature(&flat, 600, 12, 0.5, Metric::Geodesic);
        let flat_s = curvature_summary(&flat_k, 3.5);

        let mut r = lcg(9);
        let (b, span) = (250usize, 40usize);
        let n = 2 * b + span;
        let bridged = Mat::from_fn(n, 3, |i, j| {
            if i < 2 * b {
                let c = if i < b { 0.0 } else { 30.0 };
                if j == 0 { c + r() } else { r() }
            } else {
                let t = (i - 2 * b) as f64 / span as f64;
                if j == 0 { 1.0 + 28.0 * t } else { 0.5 + 0.02 * r() }
            }
        });
        let br_k = node_curvature(&bridged, n, 10, 0.5, Metric::Geodesic);
        let br_s = curvature_summary(&br_k, 3.5);

        assert!(
            asym(&br_s) > asym(&flat_s),
            "bottleneck tail ({}) should outweigh flat tail ({})",
            br_s.detail,
            flat_s.detail
        );
        // The knob is scale-free: loosening it can only admit more cells.
        let loose = curvature_summary(&br_k, 2.0);
        assert!(loose.rank >= br_s.rank, "looser cut returned fewer cells");
    }

    /// A uniform cloud samples *flat* space, whose Ricci curvature is zero.
    /// The geodesic metric must return ~0; the Euclidean one is biased
    /// positive, which is the whole reason it is not the default.
    #[test]
    fn flat_space_has_zero_curvature() {
        let mut r = lcg(4);
        let x = Mat::from_fn(400, 4, |_, _| r() + r() + r());
        let mean = |mt| {
            let k = node_curvature(&x, 400, 12, 0.5, mt);
            k.iter().sum::<f64>() / k.len() as f64
        };
        let geo = mean(Metric::Geodesic);
        assert!(geo.abs() < 0.05, "flat space gave mean kappa {geo:+.4}");
        // Documented bias: straight-line distance lets mass cut through
        // ambient space, understating transport cost and inflating kappa.
        assert!(
            mean(Metric::Euclidean) > geo + 0.05,
            "expected the Euclidean metric to read positive on flat space"
        );
    }

    /// The case the diagnostic exists for: two blobs joined by a thin bridge.
    /// The bridge cells must be negatively curved and clearly below the blob
    /// interiors. Hop distance gets this *backwards* -- it prices a filament
    /// as cheap because every step counts as 1 -- which is why the default is
    /// the geodesic and not the hop count.
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
        let k = node_curvature(&x, n, 10, 0.5, Metric::Geodesic);
        let bridge: f64 = k[2 * blob..].iter().sum::<f64>() / span as f64;
        let inside: f64 = k[..blob].iter().sum::<f64>() / blob as f64;
        assert!(bridge < 0.0, "bridge mean kappa {bridge:+.3} not negative");
        assert!(
            bridge < inside,
            "bridge {bridge:+.3} not below blob interior {inside:+.3}"
        );

        let hop = node_curvature(&x, n, 10, 0.5, Metric::Hops);
        let hop_bridge: f64 = hop[2 * blob..].iter().sum::<f64>() / span as f64;
        let hop_inside: f64 = hop[..blob].iter().sum::<f64>() / blob as f64;
        assert!(
            hop_bridge > hop_inside,
            "hop metric was expected to misrank the bridge ({hop_bridge:+.3} vs {hop_inside:+.3})"
        );
    }
}
