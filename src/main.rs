use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use scdim::{
    betti, corrdim, fiedler, geom, io, localpca, progress::Progress, rank, ricci, tangent, twonn,
};

#[derive(Parser)]
#[command(about = "Estimate the number of signal components in a single-cell matrix")]
struct Args {
    /// Input matrix (h5ad, h5seurat, 10x h5, mtx, ...).
    path: String,
    /// Genes kept, by variance of log1p(CPM). Bounds the O(g^3) eigendecomposition.
    #[arg(long, default_value_t = 2000)]
    n_genes: usize,
    /// Cells read, taken at a regular stride through the file (not a prefix --
    /// files are often ordered by tissue).
    #[arg(long, default_value_t = 20000)]
    max_cells: usize,
    /// Tracy-Widom significance level.
    #[arg(long, default_value_t = 0.001)]
    alpha: f64,
    /// Largest rank the sequential test will report.
    #[arg(long, default_value_t = 100)]
    k_max: usize,
    /// Log-normalise before biwhitening. Off by default: biwhitening is
    /// derived for counts.
    #[arg(long)]
    log: bool,
    /// Sinkhorn-Knopp under-relaxation. Lower it if biwhitening does not converge.
    #[arg(long, default_value_t = 0.3)]
    bw_damp: f64,
    /// Sinkhorn-Knopp iteration cap.
    ///
    /// 150, not rmt-spca's 1000: the residual plateaus by ~125 on every dataset
    /// tried, and rmt-spca's stall detector then burns another 100 iterations
    /// of its patience before giving up. Capping here is bit-identical output
    /// (KS, sigma2 and every rank match to 4 decimals) for a third less wall
    /// clock. Raise it if bulk-KS looks bad.
    #[arg(long, default_value_t = 150)]
    bw_max_iter: usize,
    /// Cells used by every geometric diagnostic -- TwoNN, correlation
    /// dimension, MST, Laplacian and Ricci all share one point cloud, and all
    /// are O(m^2) or worse, so it is capped.
    #[arg(long, alias = "twonn-cells", default_value_t = 2000)]
    geom_cells: usize,
    /// Top fraction of TwoNN distance ratios dropped before fitting.
    /// Near-duplicate cells produce huge ratios; this is what removes them.
    #[arg(long, default_value_t = 0.01)]
    twonn_trim: f64,
    /// Neighbours per cell in the shared k-NN graph -- the Laplacian and the
    /// Ollivier-Ricci curvature both read it. Local PCA builds its own, larger.
    #[arg(long, default_value_t = 15)]
    knn: usize,
    /// Random subsamples per decimation level in the TwoNN scale analysis.
    #[arg(long, default_value_t = 3)]
    twonn_reps: usize,
    #[arg(long, value_enum, default_value_t = Format::Txt)]
    format: Format,
    /// Ground metric for Ollivier-Ricci transport. `geodesic` (shortest path
    /// with Euclidean edge weights) is the right one; the others show why.
    #[arg(long, value_enum, default_value_t = MetricArg::Geodesic)]
    ricci_metric: MetricArg,
    /// Robust-z cut defining "extreme" curvature, in Iglewicz-Hoaglin units of
    /// the data's own kappa distribution. Scale-free: independent of k, alpha
    /// and local density, so it transfers across datasets.
    ///
    /// 2.5, not Iglewicz-Hoaglin's conventional 3.5. That 3.5 answers "which
    /// cells are outliers"; this row asks "is the left tail heavier than the
    /// right", and bottleneck cells are a percent-scale minority, not a 0.02%
    /// outlier tail. Under normality 3.5 expects 1.7 cells in 7271, so the
    /// count is noise -- it returned 0 on six of seven datasets. 2.5 expects
    /// ~45, and measured across clouds of 2000/4000/8000 the low:high ratio
    /// holds to 7% on a branching trajectory (0.68/0.67/0.72) while every blob
    /// dataset stays under 0.17, a 4x separation with no overlap.
    #[arg(long, default_value_t = 2.5)]
    ricci_cut: f64,
    /// Also measure whether distant regions of the cloud vary along the same
    /// directions: local tangent spaces compared by principal angles, against
    /// graph distance. Off by default -- it is an extra table, and it answers a
    /// question about *continua*, so it is only interesting once betti0 and
    /// fiedler have called the data one piece.
    #[arg(long)]
    tangent: bool,
    /// Suppress the stage progress on stderr.
    #[arg(long, short)]
    quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricArg {
    Geodesic,
    Hops,
    Euclidean,
}

impl From<MetricArg> for ricci::Metric {
    fn from(m: MetricArg) -> Self {
        match m {
            MetricArg::Geodesic => ricci::Metric::Geodesic,
            MetricArg::Hops => ricci::Metric::Hops,
            MetricArg::Euclidean => ricci::Metric::Euclidean,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum Format {
    Txt,
    Json,
}

/// Relative agreement required between consecutive decimation levels for them
/// to count as one plateau. ponytail: Facco reads it off the plot; the printed
/// table stays the real answer.
const PLATEAU_TOL: f64 = 0.10;

/// Bulk-KS below this counts as a good MP fit (Chardes et al. §3.1).
const KS_GOOD: f64 = 0.10;

/// Correlation-dimension slopes are noisier than TwoNN's decimation levels, so
/// the scaling range gets a looser flatness bound.
const GP_TOL: f64 = 0.15;

/// How far down the sorted MST weights to look for the patch-separating step,
/// and how big that step has to be. 2.0x says the bridging edge must be twice
/// the next-longest before "separate patches" beats "one lumpy manifold".
const MAX_PATCHES: usize = 40;
const MIN_MST_GAP: f64 = 2.0;

/// Laziness of the Ollivier-Ricci random walk: mass kept at the centre.
const RICCI_ALPHA: f64 = 0.5;

/// Run `f`, returning what it returned and how long it took. The parallel
/// stages cannot share the progress bar's single step clock.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    (f(), t.elapsed().as_secs_f64())
}

/// Peak bytes the geometry block needs per pair of cells.
///
/// Everything downstream of the cloud is O(m^2) and several of them are live at
/// once: the Gram matrix (8), the geodesic distances (4), every pairwise
/// distance sorted for the correlation integral (4), and the Laplacian's W, L
/// and eigensolver workspace (8 each). Measured rather than added up -- an
/// 8000-cell cloud peaked at 7.0 GB, which is 110 bytes per pair.
const GEOM_BYTES_PER_PAIR: usize = 110;

/// What the OS says is available, if it will say. Linux only; everywhere else
/// this returns None and the check below declines to have an opinion.
fn available_bytes() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
}

/// Refuse a cloud that will not fit before spending a minute discovering it.
///
/// The geometry block grows quadratically and has no natural cap, so
/// `--geom-cells 20000` is a 44 GB request that looks like a one-word change to
/// `--geom-cells 2000`. Being OOM-killed forty minutes into a run, after the
/// reading and biwhitening that dominate the wall clock, is the outcome worth
/// spending ten lines to avoid.
fn check_geometry_memory(m: usize) -> Result<()> {
    let need = m.saturating_mul(m).saturating_mul(GEOM_BYTES_PER_PAIR);
    let gb = |b: usize| b as f64 / 1e9;
    match available_bytes() {
        // Leave a fifth of what is free for everything else on the machine.
        Some(avail) if need > avail * 4 / 5 => anyhow::bail!(
            "a {m}-cell geometry cloud needs about {:.1} GB and only {:.1} GB is available. \
             Every geometric diagnostic is O(m^2) or worse. Lower --geom-cells (it is capped \
             at {m} here by --max-cells; the cost falls as the square, so halving it is a 4x \
             saving).",
            gb(need),
            gb(avail)
        ),
        _ => Ok(()),
    }
}

/// A JSON number, or `null` when the value is not one.
///
/// `NaN` and `Infinity` are not JSON, and `format!` will happily print them --
/// one non-finite value anywhere makes the whole line unparseable, silently,
/// in the output mode whose entire purpose is being piped into something else.
/// `slope` was already special-cased here; everything else was not.
fn num(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{x:.digits$}")
    } else {
        "null".to_string()
    }
}

/// Escape a string for a JSON string literal. Paths are user input and may
/// contain either character.
fn jstr(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Pearson correlation. Returns 0 for a degenerate input rather than NaN.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 0.0;
    }
    let (ma, mb) = (
        a[..n].iter().sum::<f64>() / n as f64,
        b[..n].iter().sum::<f64>() / n as f64,
    );
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut pr = Progress::new(!args.quiet);

    pr.begin("reading matrix");
    let counts = io::load(&args.path, args.n_genes, args.max_cells, args.log)?;
    pr.ok(
        "reading matrix",
        &format!(
            "{}x{} kept of {}x{}, {:.0}% dense",
            counts.x.nrows(),
            counts.x.ncols(),
            counts.source_shape.0,
            counts.source_shape.1,
            100.0 * counts.nnz as f64 / (counts.x.nrows() * counts.x.ncols()) as f64
        ),
    );

    pr.begin("biwhitening + eigenspectrum");
    let spec = rank::Spectrum::compute(&counts.x, args.bw_damp, args.bw_max_iter);
    pr.ok(
        "biwhitening + eigenspectrum",
        &format!("sigma2 {:.3}, bulk-KS {:.3}", spec.sigma_sq, spec.bulk_ks),
    );
    // Geometry runs in the signal subspace, not in ambient gene space: the
    // rank comes from the Tracy-Widom test, so the embedding is chosen by a
    // spectral heuristic rather than by the geometric ones themselves.
    pr.begin("tracy-widom test");
    let tw = rank::tracy_widom(&spec, args.alpha, args.k_max);
    let embed_k = tw.rank.clamp(2, spec.scores.ncols());
    let embed = spec.scores.as_ref().subcols(0, embed_k).to_owned();
    pr.ok("tracy-widom test", &format!("rank {}, embedding {embed_k}D", tw.rank));

    pr.begin("point cloud + k-NN graph");
    // Cloud::new strides by an integer, so the reachable sizes are n, n/2,
    // n/3 ... -- asking for 6000 out of 8000 rows gives 4000. Compute the size
    // the same way the constructor will, so the estimate is not a fiction.
    let stride = embed.nrows().div_ceil(args.geom_cells.max(1)).max(1);
    let m = embed.nrows().div_ceil(stride);
    check_geometry_memory(m)?;
    let cloud = geom::Cloud::new(&embed, args.geom_cells);
    let knn = cloud.knn(args.knn);
    pr.ok(
        "point cloud + k-NN graph",
        &format!("{} cells, k={}", cloud.len(), args.knn),
    );

    // The geometric diagnostics share the cloud read-only and do not talk
    // to each other, so they run concurrently. Each is internally parallel
    // too; rayon's work stealing sorts that out. Their reported times overlap.
    pr.begin("geometry (7 diagnostics, in parallel)");
    let mut mst = Default::default();
    let mut lap = Default::default();
    let mut kappa = Default::default();
    let mut gp = Default::default();
    let mut nn: (Option<rank::Estimate>, f64) = Default::default();
    let mut scale = Default::default();
    let mut lpca = Default::default();
    rayon::scope(|s| {
        s.spawn(|_| mst = timed(|| betti::mst_weights(&cloud)));
        s.spawn(|_| lap = timed(|| fiedler::laplacian_spectrum(&cloud, &knn)));
        s.spawn(|_| {
            kappa = timed(|| {
                ricci::node_curvature(&cloud, &knn, RICCI_ALPHA, args.ricci_metric.into())
            })
        });
        s.spawn(|_| gp = timed(|| corrdim::correlation_curve(&cloud, 20)));
        s.spawn(|_| nn = timed(|| Some(twonn::two_nn(&cloud, args.twonn_trim, 0.95))));
        s.spawn(|_| {
            scale = timed(|| twonn::scale_analysis(&cloud, args.twonn_reps, args.twonn_trim))
        });
        s.spawn(|_| lpca = timed(|| localpca::local_pca(&cloud, embed_k)));
    });
    pr.stage("  minimum spanning tree", mst.1, &format!("{} edges", mst.0.len()));
    pr.stage("  laplacian spectrum", lap.1, &format!("{} eigenvalues", lap.0.len()));
    pr.stage(
        "  ollivier-ricci curvature",
        kappa.1,
        &format!("{} cells, {:?} metric", kappa.0.len(), args.ricci_metric),
    );
    pr.stage("  correlation integral", gp.1, &format!("{} points", gp.0 .1));
    pr.stage("  twonn scale analysis", scale.1, &format!("{} levels", scale.0.len()));
    pr.stage("  local pca", lpca.1, &format!("{} radii", lpca.0.len()));
    pr.stage("  twonn fit", nn.1, &format!("d = {}", nn.0.as_ref().expect("spawned").rank));
    let (mst, lap, kappa, (gp, gp_n), scale, lpca) =
        (mst.0, lap.0, kappa.0, gp.0, scale.0, lpca.0);
    // Needs local-pca's dip for the tangent dimension, so it cannot join the
    // parallel block above.
    let tangent = args.tangent.then(|| {
        pr.begin("tangent-space overlap");
        let d = localpca::summary(&lpca, embed_k).rank.max(2);
        let t = timed(|| tangent::tangent_overlap(&cloud, &knn, d)).0;
        pr.ok(
            "tangent-space overlap",
            &format!("{} hop bins, d = {}, null = {:.2}", t.curve.len(), t.d, t.null),
        );
        t
    });
    pr.finish();

    let estimates = [
        tw,
        rank::mp_edge(&spec),
        nn.0.expect("spawned"),
        twonn::plateau(&scale, PLATEAU_TOL),
        corrdim::correlation_dimension(&gp, gp_n, GP_TOL),
        localpca::summary(&lpca, embed_k),
        betti::patch_count(&mst, MAX_PATCHES, MIN_MST_GAP),
        fiedler::fiedler(&lap, MAX_PATCHES),
        ricci::curvature_summary(&kappa, args.ricci_cut),
    ];

    match args.format {
        Format::Txt => {
            println!(
                "{}  {}x{} of {}x{}  q={:.4}  density={:.1}%  embed={}D  sigma2={:.4}  bulk-KS={:.4}",
                jstr(&args.path),
                spec.n,
                spec.p,
                counts.source_shape.0,
                counts.source_shape.1,
                spec.q,
                100.0 * counts.nnz as f64 / (spec.n * spec.p) as f64,
                embed_k,
                spec.sigma_sq,
                spec.bulk_ks
            );
            if spec.bulk_ks > KS_GOOD {
                println!(
                    "WARNING: bulk-KS {:.3} > {KS_GOOD}: the noise bulk does not follow \
                     Marchenko-Pastur, so tracy-widom and mp-edge below are not valid \
                     (Sinkhorn residual {:.1e})",
                    spec.bulk_ks, spec.biwhitening_residual
                );
            }
            for e in &estimates {
                // The middle column is a count; the next one is the continuous
                // reading, for the rows whose answer is a matter of degree.
                let stat = e.stat.map_or(String::new(), |v| format!("{v:.3}"));
                println!("{:<14} {:>4} {:>8}   {}", e.name, e.rank, stat, e.detail);
            }
            let lo: Vec<String> = lap.iter().take(8).map(|e| format!("{e:.3e}")).collect();
            println!("\nLaplacian spectrum (lowest 8): {}", lo.join("  "));
            println!("\nCorrelation integral (Grassberger-Procaccia):");
            println!("{:>10} {:>10} {:>8}", "r", "C(r)", "slope");
            for p in &gp {
                println!("{:>10.3} {:>10.2e} {:>8.2}", p.r, p.c, p.slope);
            }
            println!("\nTwoNN scale analysis (Facco et al. 2017):");
            println!("{:>8} {:>10} {:>8}   {}", "N", "<r2>", "d", "spread");
            for p in &scale {
                let spread = if p.spread.0 == p.spread.1 {
                    "-".to_string()
                } else {
                    format!("{:.2}-{:.2}", p.spread.0, p.spread.1)
                };
                println!(
                    "{:>8} {:>10.3} {:>8.2}   {spread}",
                    p.n, p.mean_r2, p.d
                );
            }
            if let Some(t) = &tangent {
                let (curve, null, d) = (&t.curve, t.null, t.d);
                let warn = if null > tangent::NULL_WARN * d as f64 {
                    " -- NULL IS LARGE: d/D leaves little room, read the excess column \
                     with that in mind"
                } else {
                    ""
                };
                println!(
                    "\nTangent-space overlap (d = {d} in {embed_k}D; random subspaces share \
                     {null:.2}{warn}):"
                );
                println!(
                    "{:>8} {:>9} {:>9} {:>9} {:>8}",
                    "hops", "pairs", "shared", "excess", "frac"
                );
                for p in curve.iter() {
                    let h = if p.hops >= 10 { format!(">={}", p.hops) } else { p.hops.to_string() };
                    println!(
                        "{h:>8} {:>9} {:>9.2} {:>9.2} {:>8.3}",
                        p.pairs, p.shared, p.excess, p.frac
                    );
                }
                if !t.spectrum.is_empty() {
                    // tr(M) = d, so a structureless cloud spreads the spectrum
                    // flat at d/D. That, not the pairwise d^2/D, is the chance
                    // level here.
                    println!(
                        "\nShared tangent subspace (mean projector; chance level d/D = {:.3}, \
                         concentration {:.1} of [{d}, {}]):",
                        t.chance(),
                        t.concentration(),
                        t.dim
                    );
                    let top: Vec<String> =
                        t.spectrum.iter().take(10).map(|l| format!("{l:>6.3}")).collect();
                    println!("  lambda  {}", top.join(" "));
                    // Against the null, not against d/D. d/D is where the null
                    // puts the *mean* of the spectrum; its top sits well above
                    // that for any finite ensemble, so d/D flatters lambda_1.
                    let nx: Vec<String> = t
                        .null_spectrum
                        .iter()
                        .take(10)
                        .map(|l| format!("{l:>6.3}"))
                        .collect();
                    println!("  null    {}", nx.join(" "));
                    let k = t.shared_dims();
                    println!(
                        "  shared subspace: {k} of {} directions beat their own rank's null \
                         over {} draws (p < {:.2} each)",
                        t.dim,
                        tangent::NULL_DRAWS,
                        1.0 / (tangent::NULL_DRAWS + 1) as f64
                    );

                    // Are these directions technical? Library size, ambient RNA
                    // and cell cycle all vary *inside* every region, so they are
                    // exactly what a shared-direction detector finds first. The
                    // sign of an eigenvector is arbitrary, hence |r|.
                    let depth: Vec<f64> = cloud
                        .rows
                        .iter()
                        .map(|&r| counts.totals[r].max(1.0).ln())
                        .collect();
                    let nvec = t.vectors.ncols().min(k.max(1)).min(8);
                    let rs: Vec<String> = (0..nvec)
                        .map(|c| {
                            let proj: Vec<f64> = (0..cloud.len())
                                .map(|i| {
                                    (0..t.dim)
                                        .map(|j| cloud.coords.read(i, j) * t.vectors.read(j, c))
                                        .sum()
                                })
                                .collect();
                            format!("{:.2}", pearson(&proj, &depth).abs())
                        })
                        .collect();
                    println!("  |r| of shared dirs with log total counts: {}", rs.join(" "));

                    // Which PCs the technical axis actually lives in. A global
                    // correlation says depth is present; this says where.
                    let npc = embed_k.min(12);
                    let cont = t.contamination(k);
                    let cell = |v: f64| format!("{v:>5.2}");
                    println!("\n  per-PC contamination (first {npc} PCs):");
                    println!(
                        "    {:<12}{}",
                        "PC",
                        (1..=npc).map(|j| format!("{j:>5}")).collect::<Vec<_>>().join("")
                    );
                    println!(
                        "    {:<12}{}",
                        "|r| depth",
                        (0..npc)
                            .map(|c| {
                                let v: Vec<f64> =
                                    (0..cloud.len()).map(|i| cloud.coords.read(i, c)).collect();
                                cell(pearson(&v, &depth).abs())
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    );
                    println!(
                        "    {:<12}{}",
                        "depth axis",
                        (0..npc)
                            .map(|j| cell(t.vectors.read(j, 0).powi(2)))
                            .collect::<Vec<_>>()
                            .join("")
                    );
                    println!(
                        "    {:<12}{}",
                        format!("shared({k})"),
                        (0..npc).map(|j| cell(cont[j])).collect::<Vec<_>>().join("")
                    );
                    let r1: f64 = rs.first().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    if r1 > 0.5 {
                        println!(
                            "  WARNING: the leading shared direction is library size \
                             (|r| = {r1:.2}), not biology. It is what varies inside every \
                             region. Read directions 2+ for programs, and note that cell \
                             cycle is also shared and IS biology."
                        );
                    }
                }
            }
            println!("\nLocal PCA (Little-Maggioni-Rosasco):");
            println!(
                "{:>8} {:>10} {:>8} {:>9} {:>9}",
                "k", "<r_k>", "d", "ceiling", "centres"
            );
            for p in &lpca {
                println!(
                    "{:>8} {:>10.3} {:>8.2} {:>9} {:>9}",
                    p.k, p.mean_r, p.d, p.ceiling, p.centres
                );
            }
        }
        Format::Json => {
            let ranks: Vec<String> = estimates
                .iter()
                .map(|e| {
                    let pv: Vec<String> = e
                        .pvalues
                        .iter()
                        .map(|p| {
                            if p.is_finite() {
                                format!("{p:.6e}")
                            } else {
                                "null".to_string()
                            }
                        })
                        .collect();
                    format!(
                        r#"{{"name":"{}","rank":{},"stat":{},"detail":"{}","pvalues":[{}]}}"#,
                        e.name,
                        e.rank,
                        e.stat.map_or("null".to_string(), |v| num(v, 6)),
                        jstr(&e.detail),
                        pv.join(",")
                    )
                })
                .collect();
            let head = spec.eigenvalues.iter().take(50);
            println!(
                r#"{{"path":"{}","n_cells":{},"n_genes":{},"source_shape":[{},{}],"q":{},"nnz":{},"embed_dim":{},"sigma_sq":{},"bulk_ks":{},"biwhitening_converged":{},"biwhitening_residual":{},"estimates":[{}],"eigenvalues":[{}],"laplacian_eigenvalues":[{}],"node_curvature":[{}],"correlation_integral":[{}],"scale_analysis":[{}],"local_pca":[{}],"tangent_overlap":[{}],"tangent_spectrum":[{}]}}"#,
                jstr(&args.path),
                spec.n,
                spec.p,
                counts.source_shape.0,
                counts.source_shape.1,
                num(spec.q, 8),
                counts.nnz,
                embed_k,
                num(spec.sigma_sq, 8),
                num(spec.bulk_ks, 8),
                spec.biwhitening_converged,
                num(spec.biwhitening_residual, 8),
                ranks.join(","),
                head.map(|e| num(*e, 6)).collect::<Vec<_>>().join(","),
                lap.iter().take(40).map(|e| num(*e, 8)).collect::<Vec<_>>().join(","),
                kappa.iter().map(|k| num(*k, 6)).collect::<Vec<_>>().join(","),
                                gp.iter()
                    .map(|p| format!(
                        r#"{{"r":{},"c":{},"slope":{}}}"#,
                        num(p.r, 6),
                        num(p.c, 8),
                        num(p.slope, 6)
                    ))
                    .collect::<Vec<_>>()
                    .join(","),
                scale
                    .iter()
                    .map(|p| format!(
                        r#"{{"n":{},"mean_r2":{},"d":{},"spread":[{},{}]}}"#,
                        p.n,
                        num(p.mean_r2, 6),
                        num(p.d, 6),
                        num(p.spread.0, 6),
                        num(p.spread.1, 6)
                    ))
                    .collect::<Vec<_>>()
                    .join(","),
                lpca.iter()
                    .map(|p| format!(
                        r#"{{"k":{},"mean_r":{},"d":{},"ceiling":{},"centres":{}}}"#,
                        p.k,
                        num(p.mean_r, 6),
                        num(p.d, 6),
                        p.ceiling,
                        p.centres
                    ))
                    .collect::<Vec<_>>()
                    .join(","),
                tangent.as_ref().map_or(String::new(), |t| t
                    .curve
                    .iter()
                    .map(|p| format!(
                        r#"{{"hops":{},"pairs":{},"shared":{},"excess":{},"frac":{}}}"#,
                        p.hops,
                        p.pairs,
                        num(p.shared, 6),
                        num(p.excess, 6),
                        num(p.frac, 6)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")),
                tangent.as_ref().map_or(String::new(), |t| t
                    .spectrum
                    .iter()
                    .map(|l| num(*l, 6))
                    .collect::<Vec<_>>()
                    .join(","))
            );
        }
    }
    Ok(())
}
