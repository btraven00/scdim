use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use scdim::{betti, corrdim, fiedler, geom, io, progress::Progress, rank, ricci, twonn};

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
    /// Neighbours per cell in the Laplacian k-NN graph.
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
    /// and local density, so it transfers across datasets. 3.5 is the usual
    /// outlier threshold; lower it to widen the net.
    #[arg(long, default_value_t = 3.5)]
    ricci_cut: f64,
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

/// Relative eigengap `lambda_k+1 / lambda_k` needed to call the graph split.
/// A path graph tops out around 4 (`(k+1)^2/k^2`), so 5 clears the connected
/// case that an absolute threshold on lambda_1 would misclassify.
const MIN_EIGENGAP: f64 = 5.0;

/// Laziness of the Ollivier-Ricci random walk: mass kept at the centre.
const RICCI_ALPHA: f64 = 0.5;

/// Run `f`, returning what it returned and how long it took. The parallel
/// stages cannot share the progress bar's single step clock.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    (f(), t.elapsed().as_secs_f64())
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
    let cloud = geom::Cloud::new(&embed, args.geom_cells);
    let knn = cloud.knn(args.knn);
    pr.ok(
        "point cloud + k-NN graph",
        &format!("{} cells, k={}", cloud.len(), args.knn),
    );

    // The geometric diagnostics share the cloud read-only and do not talk
    // to each other, so they run concurrently. Each is internally parallel
    // too; rayon's work stealing sorts that out. Their reported times overlap.
    pr.begin("geometry (6 diagnostics, in parallel)");
    let mut mst = Default::default();
    let mut lap = Default::default();
    let mut kappa = Default::default();
    let mut gp = Default::default();
    let mut nn: (Option<rank::Estimate>, f64) = Default::default();
    let mut scale = Default::default();
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
    pr.stage("  twonn fit", nn.1, &format!("d = {}", nn.0.as_ref().expect("spawned").rank));
    let (mst, lap, kappa, (gp, gp_n), scale) = (mst.0, lap.0, kappa.0, gp.0, scale.0);
    pr.finish();

    let estimates = [
        tw,
        rank::mp_edge(&spec),
        nn.0.expect("spawned"),
        twonn::plateau(&scale, PLATEAU_TOL),
        corrdim::correlation_dimension(&gp, gp_n, GP_TOL),
        betti::patch_count(&mst, MAX_PATCHES, MIN_MST_GAP),
        fiedler::fiedler(&lap, MAX_PATCHES, MIN_EIGENGAP),
        ricci::curvature_summary(&kappa, args.ricci_cut),
    ];

    match args.format {
        Format::Txt => {
            println!(
                "{}  {}x{} of {}x{}  q={:.4}  density={:.1}%  embed={}D  sigma2={:.4}  bulk-KS={:.4}",
                args.path,
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
                println!("{:<14} {:>4}   {}", e.name, e.rank, e.detail);
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
        }
        Format::Json => {
            let ranks: Vec<String> = estimates
                .iter()
                .map(|e| {
                    let pv: Vec<String> =
                        e.pvalues.iter().map(|p| format!("{p:.6e}")).collect();
                    format!(
                        r#"{{"name":"{}","rank":{},"detail":"{}","pvalues":[{}]}}"#,
                        e.name,
                        e.rank,
                        e.detail.replace('"', "'"),
                        pv.join(",")
                    )
                })
                .collect();
            let head = spec.eigenvalues.iter().take(50);
            println!(
                r#"{{"path":"{}","n_cells":{},"n_genes":{},"source_shape":[{},{}],"q":{},"nnz":{},"embed_dim":{},"sigma_sq":{},"bulk_ks":{},"biwhitening_converged":{},"biwhitening_residual":{},"estimates":[{}],"eigenvalues":[{}],"laplacian_eigenvalues":[{}],"node_curvature":[{}],"correlation_integral":[{}],"scale_analysis":[{}]}}"#,
                args.path,
                spec.n,
                spec.p,
                counts.source_shape.0,
                counts.source_shape.1,
                spec.q,
                counts.nnz,
                embed_k,
                spec.sigma_sq,
                spec.bulk_ks,
                spec.biwhitening_converged,
                spec.biwhitening_residual,
                ranks.join(","),
                head.map(|e| format!("{e:.6}")).collect::<Vec<_>>().join(","),
                lap.iter()
                    .take(40)
                    .map(|e| format!("{e:.8}"))
                    .collect::<Vec<_>>()
                    .join(","),
                kappa.iter().map(|k| format!("{k:.6}")).collect::<Vec<_>>().join(","),
                                gp.iter()
                    .map(|p| format!(
                        r#"{{"r":{:.6},"c":{:.8},"slope":{:.6}}}"#,
                        p.r, p.c, if p.slope.is_finite() { p.slope } else { 0.0 }
                    ))
                    .collect::<Vec<_>>()
                    .join(","),
                scale
                    .iter()
                    .map(|p| format!(
                        r#"{{"n":{},"mean_r2":{:.6},"d":{:.6},"spread":[{:.6},{:.6}]}}"#,
                        p.n, p.mean_r2, p.d, p.spread.0, p.spread.1
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }
    Ok(())
}
