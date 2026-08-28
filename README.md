# scdim

How many components should you keep? This crate reads a single-cell count
matrix and reports what several rank-selection heuristics think, so you can see
where they agree and where they don't. It does not pick for you — when the
heuristics disagree, that disagreement is the finding.

I/O comes from [`scx`](../scx) (h5ad, h5seurat, 10x h5, mtx, npy). Biwhitening
and Marchenko-Pastur theory come from [`rmt-spca`](../rmt-spca).

## Usage

```bash
cargo run --release -- data.h5ad
cargo run --release -- data.h5ad --n-genes 2000 --alpha 0.01 --format json
```

```
data.h5ad  3000x1000 of 5000x27998  q=0.3333  sigma2=0.7809
tracy-widom      37   stopped at component 38: TW1 statistic 1.916, p = 1.198e-2 > 0.001
mp-edge          37   eigenvalues above lambda+ = 2.4880
twonn            88   d = 88.46 (95% CI 84.07-93.07), 1485/1500 ratios after 1% trim
twonn-plateau     0   no plateau: d drifts 88.5 -> 58.3 from N=1500 to N=93, no 3 consecutive levels within 10%
corr-dim         17   D = 16.53 over 6 scales flat to 15% -- AT THE CEILING: 1659 points support D <= 6.4

TwoNN scale analysis (Facco et al. 2017):
       N       <r2>        d   spread
    1500     38.279    88.46   -
     750     38.683    80.19   77.58-81.62
     375     39.102    72.79   68.11-81.28
     187     39.664    61.07   49.60-71.01
      93     40.535    58.29   50.38-68.59
```

`--format json` emits the same numbers on one line, plus the top 50 eigenvalues
and the per-component Tracy-Widom p-value sequence:

```
"pvalues": [0.0, 0.0, ..., 6.39e-5, 5.71e-4, 4.01e-3]
```

Exact zeros are p < 1e-12, the floor of the Painlevé II integration; anything
above that is accurate.

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-genes` | 2000 | Genes kept, by variance of log1p(CPM). Bounds the O(g³) EVD. |
| `--max-cells` | 20000 | Cells read from the top of the file. |
| `--alpha` | 0.001 | Tracy-Widom significance level. |
| `--k-max` | 100 | Largest rank the sequential test will report. |
| `--log` | off | Log-normalise before biwhitening (biwhitening is derived for counts). |
| `--bw-damp` | 0.3 | Sinkhorn-Knopp under-relaxation. Lower it if biwhitening won't converge. |
| `--bw-max-iter` | 150 | Sinkhorn-Knopp iteration cap. |
| `--twonn-cells` | 2000 | Cells used for the TwoNN neighbour search (it is O(m²g)). |
| `--twonn-trim` | 0.01 | Top fraction of TwoNN distance ratios dropped before fitting. |
| `--twonn-reps` | 3 | Random subsamples per decimation level in the scale analysis. |

## The pipeline

1. Stream the matrix, keep the top-variance genes, densify.
2. **Biwhiten** (Sinkhorn-Knopp, `rmt_spca::biwhitening`) so every gene and cell
   has unit variance. This is the load-bearing step: raw counts are
   heteroskedastic and their bulk spectrum is *not* Marchenko-Pastur, so none of
   the thresholds below mean anything without it.
3. Mean-centre, form the covariance, take the full eigenspectrum.
4. Estimate σ² by matching the bulk median to the MP median — robust to the very
   outliers we are about to count — and divide it out.
5. Run each heuristic on the resulting spectrum.

The header line carries the diagnostic that decides whether any of it means
anything: **bulk-KS**, the Kolmogorov-Smirnov distance between the bulk
eigenvalues and the MP CDF (Chardès et al. §3.1, ≤ 0.10 is a good fit). If that
is large the noise bulk is not Marchenko-Pastur and the spectral heuristics are
void, whatever numbers they print.

Note what is *not* the diagnostic: Sinkhorn-Knopp's own convergence flag. It
comes back false on every real dataset tried here, and it is measuring the wrong
thing — under damping the iteration settles into an oscillation, so its
step-to-step residual plateaus and never reaches `tol`. Running 20000 iterations
instead of 1000 gives a bit-identical residual, and σ² and the ranks are stable
to three decimals across a 6× change in `--bw-damp`. Meanwhile bulk-KS on the
same data is 0.03–0.10, i.e. the fit is fine. The flag is still reported, but it
is not evidence.

## Heuristics

### 1. Tracy-Widom (implemented)

Under the null "everything left is noise", the largest eigenvalue of a real
Wishart matrix — centred and scaled by Johnstone (2001) with Ma's (2012) −½
refinement — converges to the order-1 Tracy-Widom law:

```
mu    = (√(n−½) + √(p−½))²
sigma = (√(n−½) + √(p−½)) · (1/√(n−½) + 1/√(p−½))^⅓
(n·λ − mu) / sigma  →  TW₁
```

Test λ₁; if it exceeds the upper-tail quantile it is signal, so drop it, shrink
(n, p) by one and test λ₂. The rank is the number rejected before the first
acceptance — the same sequential construction as Patterson, Price & Reich (2006)
in `smartpca`.

**p-values are exact**, not read off a quantile table. `tw::tw1_sf` integrates
Painlevé II directly:

```
F₂(s) = exp(−I(s)),        I(s) = ∫ₛ^∞ (x−s) q(x)² dx
F₁(s) = √(F₂(s)·exp(−K(s))), K(s) = ∫ₛ^∞ q(x) dx
```

with q the Hastings-McLeod solution of q″ = sq + 2q³, q(s) ~ Ai(s) as s → ∞.
RK4 backwards from s = 12 (the stable direction — the unwanted Bi-like solution
decays as you step down), initial condition from the Airy asymptotic expansion,
and `expm1` for the tail so p-values below 1e-16 stay accurate instead of
rounding to zero. It reproduces the published TW₁ quantiles to <5e-4; that check
is the unit test.

### 2. Marchenko-Pastur edge

Count eigenvalues above λ₊ = (1 + √q)². This is the asymptotic limit of the same
idea with no finite-size correction, so it is systematically more permissive.
It's here to be compared against: when TW and MP disagree by a lot, suspect the
biwhitening rather than the biology.

### 3. TwoNN

Intrinsic dimension by the ratio of the first two nearest-neighbour distances
(Facco et al. 2017), following `intRinsic::twonn_mle`. For each cell,
μᵢ = r₂/r₁ is Pareto(1, d) when the density is locally constant, so

```
d̂ = (n − 1) / Σᵢ log μᵢ
```

with a Gamma(n, n−1) sampling law giving the confidence interval, and the
reference's 1% upper trim (a single pair of near-duplicate cells produces a huge
μ and drags the estimate down). Distances come from the Gram matrix, so the
neighbour search is one BLAS-3 call.

This answers a **different question** from the two above: the dimension of the
manifold the cells lie on, not the number of linear components above noise. Read
it accordingly —

- It runs on the PCA scores truncated at the Tracy-Widom rank, not on the
  ambient gene space — see "Geometry runs in the signal subspace" below for why
  that is not optional.

### 4. TwoNN scale analysis / decimation (implemented)

TwoNN only ever looks at the first two neighbours, so it always probes the
smallest scale available. Facco et al. give the fix (§"Estimating the intrinsic
dimension of a dataset", Fig. 3): shrink the sample, which pushes the typical
neighbour distance out as N^(−1/d), and watch d̂ move.

> "The relevant ID of the dataset can be obtained by finding a range of N for
> which d̂(N) is constant, and thus a plateau in the graph of d̂(N). The value of
> d at the plateau is the number of 'soft', or relevant, directions in the
> dataset."

The implementation halves the sample repeatedly down to N = 50, draws
`--twonn-reps` random subsamples per level (deterministically seeded — a
diagnostic you can't rerun isn't one), and reports the mean d̂ with the observed
min/max across replicates. That spread is an empirical range, **not** a
confidence interval. Cost is about 2× a single TwoNN fit, since the halved
levels are quadratically cheaper.

`twonn-plateau` looks for the longest run of consecutive levels all within 10%
of each other. The bound is on the whole run, not on adjacent steps: a steady
9%-per-level decline passes an adjacent-step test while halving over five
levels, and calling that a plateau is precisely the error to avoid. It needs at
least three levels — two adjacent estimates landing within 10% happens all the
time on a monotone curve.

Two things to expect on real single-cell data, both visible in the example above:

- **There is often no plateau.** d̂ falls monotonically (88 → 58) and the tool
  says so rather than inventing a number. That is a result: at these scales the
  cloud does not look like a manifold of any fixed dimension.
- **Decimation has little leverage when d̂ is large.** The scale probed goes as
  N^(−1/d), so with d̂ ≈ 88 a 16× decimation moves ⟨r₂⟩ by 6% (38.3 → 40.5).
  Most of the change in d̂ across levels is therefore a small-sample effect on
  the estimator, not a genuine walk up the scale axis. Facco's own examples have
  d of order 10, where the lever actually works. Read the table with that in
  mind — and if you need real scale resolution, the ratio-based Gride estimator
  (Denti et al. 2022) varies the neighbour order instead of the sample size and
  does not have this problem.

### 5. Correlation dimension, Grassberger-Procaccia (implemented)

Count the pairs closer than r:

```
C(r) = 2/(N(N−1)) · #{i<j : ‖xᵢ−xⱼ‖ < r},   C(r) ~ r^D
```

so D is the slope of log C against log r. Where TwoNN fits a parametric law to
one neighbour ratio, this reads the scaling of the whole pair-distance
distribution — which is why it is the better instrument for continuous,
branching structure (a differentiation or hematopoiesis trajectory) rather than
a set of discrete clusters. It will never return "the number of cell types"; it
returns the dimension of the set the cells trace out.

Scales are placed so C is log-spaced from 1e-4 to 0.3, which gives every local
slope a comparable number of pairs and keeps all of them out of the saturated
region where C → 1 forces the slope to zero. `corr-dim` reports the mean over
the longest run of slopes flat to 15%, using the same whole-run criterion as the
TwoNN plateau.

**The sample-size ceiling is not a footnote.** Estimating D from N points needs
roughly D ≤ 2 log₁₀ N (Eckmann & Ruelle 1992) — with 1659 points, D ≤ 6.4. Above
that the correlation integral has no scaling range left and the estimator
returns something too small *without failing*, which is the dangerous kind of
wrong. So the ceiling is printed next to every estimate and the row shouts when
the estimate is within 80% of it:

```
corr-dim         17   D = 16.53 over 6 scales flat to 15% -- AT THE CEILING: 1659
                      points support D <= 6.4 (Eckmann-Ruelle), so this is a lower
                      bound, not a measurement
```

On whole-atlas data that is the expected outcome: the cloud is far too
high-dimensional for GP at any sample size you can afford (the ceiling grows as
log N, so 10⁶ cells buys D ≤ 12). Where it earns its keep is exactly the case it
was suggested for — a trajectory dataset with D of order 2–5, comfortably under
the ceiling. The unit tests cover that regime: a filled 3-cube in R²⁰ returns 3,
a 1-D curve in R¹⁰ returns 1.

### 6. Candidates not yet implemented

- **Parallel analysis / permutation.** Shuffle each gene independently, recompute
  the spectrum, keep components above the permuted λ₁ (or its 95th percentile
  over B permutations). Makes no distributional assumption at all, which makes it
  the right cross-check on whether biwhitening actually worked. Costs B full EVDs.
- **Cross-validated / bi-cross-validation reconstruction error** (Owen & Perry
  2009). Hold out a submatrix, reconstruct it from a rank-k fit of the rest, pick
  the k minimising held-out error. Optimises for the thing you usually want
  (prediction) rather than for a null hypothesis, and needs no noise model.
- **Eigenvalue-gap / elbow.** Largest ratio λ_k/λ_{k+1}, or the knee of the scree
  curve. Cheap, no theory, and worth reporting precisely because it is what
  everyone does by eye.
- **BBP overlap floor.** `rmt_spca::rmt::RmtTheory::predicted_overlap` already
  gives the expected cos²θ between an outlier eigenvector and the true direction.
  Instead of "is this above noise?", ask "is this direction *estimable*?" and cut
  where the predicted overlap drops below, say, 0.5. Different and often smaller
  answer than TW.
- **Other intrinsic-dimension estimators** (Levina-Bickel MLE, correlation
  dimension, Gride, the heterogeneous-ID mixture). Same family as TwoNN above,
  different bias/variance trade-offs. See the survey: Binnie, Dłotko,
  Harvey, Malinowski & Yim, *A Survey of Dimension Estimation Methods*,
  [arXiv:2507.13887](https://arxiv.org/abs/2507.13887v1), which compares them on
  robustness to hyperparameters, sample size, and non-linear geometry.

## References

- Chardès et al., *A statistical physics approach to characterise single-cell
  data*, [arXiv:2509.15429](https://arxiv.org/abs/2509.15429) — biwhitening + MP.
- Johnstone, *On the distribution of the largest eigenvalue in principal
  components analysis*, Ann. Statist. 29 (2001).
- Ma, *Accuracy of the Tracy-Widom limits for the extreme eigenvalues in white
  Wishart matrices*, Bernoulli 18 (2012).
- Patterson, Price & Reich, *Population structure and eigenanalysis*, PLoS Genet.
  2 (2006).
- Grassberger & Procaccia, *Measuring the strangeness of strange attractors*,
  Physica D **9**, 189 (1983) — the correlation integral.
- Eckmann & Ruelle, *Fundamental limitations for estimating dimensions and
  Lyapunov exponents in dynamical systems*, Physica D **56**, 185 (1992) — the
  D ≤ 2 log₁₀ N ceiling.
- Facco, d'Errico, Rodriguez & Laio, *Estimating the intrinsic dimension of
  datasets by a minimal neighborhood information*, Sci. Rep. **7**, 12140 (2017)
  — TwoNN and the decimation/plateau analysis.
- Denti, Doimo, Laio & Mira, *The generalized ratios intrinsic dimension
  estimator*, Sci. Rep. **12**, 20005 (2022) — Gride, the scale analysis done by
  varying neighbour order rather than sample size.
- Denti, `intRinsic`: An R Package for Intrinsic Dimension Estimation,
  J. Stat. Softw. (2023) — the reference implementation `twonn_mle` follows.
- Binnie et al., *A Survey of Dimension Estimation Methods*,
  [arXiv:2507.13887](https://arxiv.org/abs/2507.13887v1).

## What these heuristics do on labelled data

Six datasets with ground-truth cell-type labels, all at `--n-genes 2000
--max-cells 5000`:

| dataset | K (labels) | bulk-KS | σ² | TW | MP | TwoNN |
|---------|-----------|---------|-----|----|----|-------|
| tm-droplet-trachea | 5 | 0.031 | 0.74 | 77 | 78 | 83 |
| pbmc (raw) | 11 | 0.056 | 0.71 | 80 | 80 | 108 |
| tm-facs-marrow | 22 | 0.031 | 0.65 | 84 | 88 | 82 |
| pbmc | 31 | 0.048 | 0.74 | 71 | 72 | 119 |
| tm-droplet | 55 | 0.095 | 0.57 | 184 | 176 | 47 |
| tm-facs | 81 | 0.081 | 0.59 | 172 | 166 | 41 |

Read it as: **the TW rank does not track the label count.** K goes 5 → 31 while
the rank sits flat at 71–88; the jump to ~170 comes with the two whole-organism
atlases, not with more labels. What it is tracking is tissue breadth. That is
not a failure of the estimator — cell-type labels are a coarse discretisation of
the signal subspace, which also contains within-type variation, batch, cell
cycle and depth — but it does mean "number of clusters" is the wrong yardstick
for it, and anyone reaching for these numbers to pick a PCA rank should know
that the answer is not K.

bulk-KS is ≤ 0.10 throughout, so the MP assumption holds on all six and the
ranks are valid; σ² lands at 0.57–0.74 rather than 1, which is the honest
statement that biwhitening under-scales, and is corrected for.

One caveat on how these were measured: `--max-cells` takes cells at a stride
through the file, never a prefix. Tabula Muris is ordered by tissue, so the
first 5000 of its 44779 cells cover 8 of the 81 types and give quite different
answers (TW 130, TwoNN 90).

## Structure: continuous manifold or discrete patches?

`betti0` answers this, and it is cheaper than it looks. The 0-dimensional
persistent homology of a Vietoris-Rips filtration **is** single-linkage
clustering, and single-linkage **is** the Euclidean minimum spanning tree: the
H₀ barcode's death times are exactly the MST edge weights, so

```
β₀(r) = N − #{MST edges ≤ r}
```

is the whole barcode. Prim's algorithm over the Gram matrix the other geometric
heuristics already build gives it in O(N²). No simplicial complex is
constructed and no persistence library is needed — for H₀ one would compute this
and stop.

Sort the weights. On a connected manifold they are unimodal and the largest is
unremarkable. On separated patches, the few edges bridging them are much longer
than the rest and the sorted weights show a step:

```
gap = max over the upper tail of w[i+1] / w[i]
```

The number of edges above the step, plus one, is the patch count. Same reasoning
as the Laplacian eigengap, but with no k-NN graph, no choice of k and no sparse
eigensolver — and unlike the Fiedler value it does not assume the answer is two.
Ambient RNA and doublets are precisely what bridges real patches, which is why
the statistic is the *ratio* at the step rather than a count of zero
eigenvalues: the bridges make the components formally connected while leaving
the step intact.

Measured (5000 cells, 2000 genes, geometry in the signal subspace):

| dataset | K | largest MST step | longest edge / median | verdict |
|---|---|---|---|---|
| tm-droplet-trachea | 5 | 1.10× | 1.7× | continuous |
| pbmc | 31 | 1.37× | 2.7× | continuous |
| tm-facs | 81 | 1.13× | 2.2× | continuous |

All three read continuous, and that is the right answer rather than a failure:
single-linkage on real scRNA-seq essentially never separates, which is why the
field clusters k-NN graphs with Leiden instead. The useful output is the
statistic, not the verdict — pbmc is measurably the most patchy of the three,
which matches immune types being more discrete than a whole-organism atlas.

## Geometry runs in the signal subspace, not in gene space

TwoNN, correlation dimension and Betti-0 all run on the PCA scores truncated at
the Tracy-Widom rank, not on the 2000-gene biwhitened matrix. In 2000 ambient
dimensions distances concentrate hard enough to destroy the measurement: on
Tabula Muris FACS the longest MST edge was 1.2× the median and the largest
single-linkage step 1.01×, i.e. **81 well-separated cell types were
indistinguishable from a smooth curve**. Projecting first moves those to 2.2×
and 1.13×, and TwoNN's estimate from 41 to 13.

The eigenvectors come free with the EVD that is run anyway, so this costs
nothing. The rank comes from a *spectral* heuristic, so the geometric ones are
not choosing their own embedding.

## Deferred: scwarp acceleration

Sinkhorn-Knopp is ~70% of the run at 20k cells (12.1 s of 17.1 s) and this path
is dense. `../scwarp` branch `feat/biwhitening` has a sparse CSR Sinkhorn-Knopp
(`scwarp_core::biwhiten::sinkhorn_knopp`, CPU + wgpu + CUDA) plus its own MP
edge/rank/KS diagnostics. Swap to it when that branch reaches `main` — it is
currently 10 commits ahead of `main` and 3 behind, so a path dependency on
`../scwarp` does not see it. Three things measured before switching, so nobody
repeats the work:

1. **GPU is not the lever.** scwarp's own profile has its dense GPU kernel
   losing to its CPU sparse loop at every size, and the sparse GPU winning only
   ~14% at 20k×2k once the one-time setup is counted. Their CUDA entry point
   runs Sinkhorn on the CPU for this reason.
2. **Sparsity is not the lever here either.** Top-variance gene selection keeps
   the *dense* genes — the matrix fed to Sinkhorn is 49.6% dense, where sparse
   and dense cost about the same. It becomes the lever only if whitening moves
   to the full gene set (~7% dense), which is what scwarp argues for anyway.
3. **The iteration count is the lever.** scwarp uses Landa et al.'s plain
   Sinkhorn on the variance matrix and converges in 20–35 iterations; the
   Chardès variant used here oscillates under damping and needs ~225.

Two methodological disagreements to settle before adopting it: scwarp does not
mean-centre after whitening (it argues centring shifts the spectrum off the MP
law this all rests on), and it whitens the whole gene set rather than an HVG
subset.

Already taken, independent of scwarp: `--bw-max-iter` defaults to 150 rather
than rmt-spca's 1000. The residual plateaus by ~125 on every dataset tried and
rmt-spca's stall detector then burns 100 more iterations of patience; capping
gives bit-identical output (KS, σ² and every rank match to 4 decimals) for a
third less wall clock.
