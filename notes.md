# Notes

Method detail, measurements, and the reasoning behind the defaults. The README
is the user-facing part; this is the working record.

## Contents

- [Tracy-Widom](#tracy-widom)
- [Marchenko-Pastur edge](#marchenko-pastur-edge)
- [Why Sinkhorn's convergence flag is not the diagnostic](#why-sinkhorns-convergence-flag-is-not-the-diagnostic)
- [TwoNN](#twonn) and [the scale analysis](#twonn-scale-analysis-decimation)
- [Correlation dimension](#correlation-dimension)
- [Local PCA](#local-pca)
- [Betti-0 from the MST](#betti-0-from-the-mst)
- [Fiedler value and eigengap](#fiedler-value-and-eigengap)
- [Ollivier-Ricci curvature](#ollivier-ricci-curvature)
- [One cloud, six diagnostics](#one-cloud-six-diagnostics)
- [Geometry in the signal subspace](#geometry-in-the-signal-subspace)
- [Does a bigger cloud help?](#does-a-bigger-cloud-help)
- [Measurements on labelled data](#measurements-on-labelled-data)
- [Deferred: scwarp acceleration](#deferred-scwarp-acceleration)
- [Candidates not implemented](#candidates-not-implemented)

## Tracy-Widom

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
acceptance — the sequential construction of Patterson, Price & Reich (2006) in
`smartpca`. One deviation from theirs: they renormalise the remaining
eigenvalues to their own mean at each step; here σ² is estimated once, by
median-matching the whole bulk against MP, which is more robust to the outliers
being counted.

p-values are integrated, not read off a quantile table. `tw::tw1_sf` solves
Painlevé II:

```
F₂(s) = exp(−I(s)),          I(s) = ∫ₛ^∞ (x−s) q(x)² dx
F₁(s) = √(F₂(s)·exp(−K(s))), K(s) = ∫ₛ^∞ q(x) dx
```

with q the Hastings-McLeod solution of q″ = sq + 2q³, q(s) ~ Ai(s) as s → ∞.
RK4 backwards from s = 12 — the stable direction, since the unwanted Bi-like
solution decays as you step down — initial condition from the Airy asymptotic
expansion, and `expm1` for the tail so small p-values stay accurate instead of
rounding to zero. It reproduces the published TW₁ quantiles to < 5e-4, which is
the unit test. Exact zeros in the JSON output are p < 1e-12, the floor of the
integration.

Caveats worth keeping in mind:

- p-values are raw. The sequential stop is the only multiplicity control, in
  the sense that only the first acceptance is acted on.
- The rank is a function of n. Detection power grows with the number of cells,
  so `--max-cells` moves the answer; datasets are only comparable at equal n.
- Genes are pre-selected by variance before the spectrum is taken, which is
  selection on the same statistic. The MP null is then approximate and the rank
  is biased slightly high. Quantifying that would need a permutation control
  (see [candidates](#candidates-not-implemented)).

## Marchenko-Pastur edge

Count eigenvalues above λ₊ = (1 + √q)². The asymptotic limit of the same idea
with no finite-size correction, so it is systematically more permissive. It is
here to be compared against: a large TW/MP disagreement points at the
biwhitening, not at the biology.

## Why Sinkhorn's convergence flag is not the diagnostic

The flag comes back false on every real dataset tried here, and it measures the
wrong thing: under damping the iteration settles into an oscillation, so the
step-to-step residual plateaus and never reaches `tol`. Running 20000
iterations instead of 1000 gives a bit-identical residual, and σ² and the ranks
are stable to three decimals across a 6× change in `--bw-damp`. bulk-KS on the
same data is 0.03–0.10, i.e. the fit is fine. The flag is reported but is not
evidence.

`--bw-max-iter` therefore defaults to 150 rather than rmt-spca's 1000: the
residual plateaus by ~125 on every dataset tried and rmt-spca's stall detector
then burns another 100 iterations of patience. Capping gives bit-identical
output (KS, σ² and every rank match to 4 decimals) for a third less wall clock.

## TwoNN

Intrinsic dimension from the ratio of the first two nearest-neighbour distances
(Facco et al. 2017), following `intRinsic::twonn_mle`. For each cell,
μᵢ = r₂/r₁ is Pareto(1, d) when the density is locally constant, so

```
d̂ = (n − 1) / Σᵢ log μᵢ
```

with a Gamma(n, n−1) sampling law giving the confidence interval, and the
reference's 1% upper trim — a single pair of near-duplicate cells produces a
huge μ and drags the estimate down. Distances come from the Gram matrix, so the
neighbour search is one BLAS-3 call.

This answers a different question from the spectral rows: the dimension of the
manifold the cells lie on, not the number of linear components above noise.

## TwoNN scale analysis (decimation)

TwoNN only looks at the first two neighbours, so it always probes the smallest
scale available. Facco et al. give the fix (Fig. 3): shrink the sample, which
pushes the typical neighbour distance out as N^(−1/d), and watch d̂ move.

> "The relevant ID of the dataset can be obtained by finding a range of N for
> which d̂(N) is constant, and thus a plateau in the graph of d̂(N). The value of
> d at the plateau is the number of 'soft', or relevant, directions in the
> dataset."

The implementation halves the sample down to N = 50, draws `--twonn-reps`
subsamples per level (deterministically seeded), and reports the mean d̂ with
the observed min/max across replicates. That spread is an empirical range, not
a confidence interval. Cost is about 2× a single TwoNN fit, since the halved
levels are quadratically cheaper.

`twonn-plateau` takes the longest run of consecutive levels all within 10% of
each other. The bound is on the whole run, not on adjacent steps: a steady
9%-per-level decline passes an adjacent-step test while halving over five
levels. Three levels minimum — two adjacent estimates landing within 10%
happens all the time on a monotone curve. Ties go to the first (largest-N) run.

Two things to expect on real data:

- **There is often no plateau.** d̂ falls monotonically and the tool says so
  rather than inventing a number. At these scales the cloud does not look like
  a manifold of fixed dimension.
- **Decimation has little leverage when d̂ is large.** The scale probed goes as
  N^(−1/d), so at d̂ ≈ 88 a 16× decimation moves ⟨r₂⟩ by 6%. Most of the change
  in d̂ across levels is then a small-sample effect on the estimator, not a walk
  up the scale axis. Facco's own examples have d of order 10, where the lever
  works. For real scale resolution, Gride (Denti et al. 2022) varies the
  neighbour order instead of the sample size and does not have this problem.

Worked example, N = 1866, ceiling 2·log₁₀N = 6.5:

```
    1866   13.966   13.26
     933   15.176   10.54
     466   17.122    8.23
     233   19.488    6.88   <- flat
     116   22.330    6.80   <- flat
      58   25.231    7.20   <- flat  =>  twonn-plateau d ~ 7
```

with a correlation integral whose slope falls 9.9 → 2.95 (at r ≈ 19) and rises
back to 6.4. Read together: TwoNN plateaus at d ≈ 7, and ⟨r₂⟩ grew 1.8× over a
32× decimation, which is what d ≈ 7 predicts (32^{1/7} = 1.63), so the lever
worked and the plateau is real. GP's ceiling is 6.5 — it cannot resolve 7, and
its minimum of ~3 is the documented low-side bias above the ceiling rather than
a contradiction. The rise at large r says the density is uneven; cross-check
against `betti0` and `fiedler`, and if those say connected then it is uneven
density, not separated patches.

## Correlation dimension

```
C(r) = 2/(N(N−1)) · #{i<j : ‖xᵢ−xⱼ‖ < r},   C(r) ~ r^D
```

so D is the slope of log C against log r. Where TwoNN fits a parametric law to
one neighbour ratio, this reads the scaling of the whole pair-distance
distribution, which makes it the better instrument for continuous branching
structure rather than discrete clusters. It will never return "the number of
cell types"; it returns the dimension of the set the cells trace out.

Scales are placed so C is log-spaced from 1e-4 to 0.3 — every local slope then
rests on a comparable number of pairs, and none sit in the saturated region
where C → 1 forces the slope to zero. `corr-dim` reports the mean over the
longest run of slopes flat to 15%, same whole-run criterion as the TwoNN
plateau.

The sample-size ceiling is the thing to watch: estimating D from N points needs
roughly D ≤ 2·log₁₀N (Eckmann & Ruelle 1992), so 1659 points give D ≤ 6.4.
Above that the correlation integral has no scaling range left and the estimator
returns something too small without failing, which is the dangerous kind of
wrong. The ceiling is printed next to every estimate and the row shouts when
the estimate is within 80% of it.

On whole-atlas data hitting the ceiling is the expected outcome — the cloud is
too high-dimensional for GP at any affordable sample size, and the ceiling
grows as log N, so 10⁶ cells buys only D ≤ 12. Where it earns its keep is the
trajectory case, D of order 2–5. The unit tests cover that regime: a filled
3-cube in R²⁰ returns 3, a 1-D curve in R¹⁰ returns 1.

Note the flat-run search can settle on the small-r noise floor when that is the
longest flat stretch, which is what the ceiling warning is for.

## Local PCA

`tracy-widom` is the rank of the covariance of the whole cloud; `twonn` is the
dimension of the tangent space at a point. They are the same quantity at two
scales — the intrinsic dimension is the rank of the *local* covariance as
r → 0 — so eigendecomposing each cell's k-neighbourhood and growing k walks
continuously from one row to the other. Little, Maggioni & Rosasco (2017).

In a ball of radius r on a d-manifold the local covariance has d eigenvalues of
order r², the tangent directions, and up to d(d+1)/2 of order r⁴ from the second
fundamental form. A flat patch gives the same d at every radius; a curved one
spends extra linear dimensions as the ball grows. **The gap between rows 1 and 3
is not error, it is bending.**

Read off the double-centred squared-distance matrix of the neighbourhood
(classical MDS) rather than the coordinates — the same "XᵀX and XXᵀ share their
nonzero eigenvalues" identity `rank` uses, and it needs nothing from the cloud
but `d2`. Centres are strided to 256; radii double from k = 16 to k = 256.

### No threshold, and the bias that costs

Counting "eigenvalues above the r⁴ shoulder" needs a cut, and every cut would be
one more knob calibrated on one dataset. The participation ratio needs none:

```
d = (Σλ)² / Σλ²
```

exactly d for an isotropic d-dimensional patch. Two things about it:

- **It is deflated at small k.** A covariance from k+1 points has dispersed
  sample eigenvalues even when the population ones are equal. For Wishart,
  E[tr S] = d and E[tr S²] = d(d+k+1)/k, so the ratio converges to d/(1 + d/k),
  not to d — 2.2 instead of 3 at k = 8, measured, matching the prediction of
  2.18. Inverting gives `d = PR / (1 − PR/k)`, and that is the whole correction.
  It is largest exactly where the number matters most, at the small-radius end.
- **It weights by variance, so it is conservative.** A single weak direction
  barely moves it: a 90° arc of a circle has λ₂/λ₁ = 4% and still reads 1.09. It
  will not manufacture dimensions out of mild curvature and it will not resolve
  them either. What it does see is directions carrying *comparable* variance,
  which is the noise case and the multiple-cell-types case.

### Read the dip

Same three regimes as C(r), for the same reason: at small k the ball is inside
the noise, which is full-rank, so d is inflated; at large k it swallows
curvature and neighbouring cell types, so d is inflated again; the manifold is
the minimum between. Both ends are failure modes and the row names which one it
is on.

A level pinned against `min(k, embed)` is excluded from the dip search. It has
measured nothing, and it is pinned *low*, so leaving it in makes it the minimum
by construction — on the Seurat PBMC file, k = 16 and k = 32 both sat on their
ceilings (16.00 and 31.98) and the summary reported the ceiling back as the
answer until they were filtered out.

Measured, 4000 cells:

| dataset | embed | twonn | twonn-plateau | corr-dim | local-pca dip |
|---|---|---|---|---|---|
| Norman 2019 | 29 | 17.70 | 13.78 | 15.1 (ceiling) | **14.23** at k=256 |
| zheng2017 pbmc | 20 | 13.72 | none (13.7→8.3) | 11.8 (ceiling) | 8.85 at k=256 |
| seurat pbmc | 76 | 25.19 | none (25.2→14.0) | 17.8 (ceiling) | 20.56 at k=256 |

Norman is the case worth reading: `twonn-plateau` and `local-pca` are two
estimators with unrelated failure modes and they land within 0.5 of each other,
against a raw TwoNN of 17.7 and a `corr-dim` that is above its own ceiling and
therefore meaningless. On the other two both rows say the same thing — still
falling at the largest radius, every scale reachable is noise-dominated — which
is a verdict, not a number.

The ladder stops at k = 256 (`K_CAP`), not at the cloud size. All three
datasets are still descending there, so the honest reading is an upper bound.
Raising the cap is O(k³) per centre and would need the centre count to drop to
pay for it.

## Betti-0 from the MST

The 0-dimensional persistent homology of a Vietoris-Rips filtration is
single-linkage clustering, and single-linkage is the Euclidean minimum spanning
tree: the H₀ barcode's death times are the MST edge weights, so

```
β₀(r) = N − #{MST edges ≤ r}
```

is the whole barcode. Prim's algorithm over the Gram matrix the other geometric
heuristics already build gives it in O(N²). No simplicial complex, no
persistence library — for H₀ one would compute this and stop.

Sort the weights. On a connected manifold they are unimodal and the largest is
unremarkable. On separated patches the few bridging edges are much longer and
the sorted weights show a step; `gap = max over the upper tail of w[i+1]/w[i]`,
and the number of edges above the step plus one is the patch count. Same
reasoning as the Laplacian eigengap, but with no k-NN graph, no choice of k and
no sparse eigensolver — and unlike the Fiedler value it does not assume the
answer is two.

Ambient RNA and doublets are what bridges real patches, which is why the
statistic is the ratio at the step rather than a count of zero eigenvalues: the
bridges make the components formally connected while leaving the step intact.

Measured (5000 cells, 2000 genes, geometry in the signal subspace):

| dataset | K | largest MST step | longest edge / median | verdict |
|---|---|---|---|---|
| tm-droplet-trachea | 5 | 1.10× | 1.7× | continuous |
| pbmc | 31 | 1.37× | 2.7× | continuous |
| tm-facs | 81 | 1.13× | 2.2× | continuous |

All three read continuous, which is the right answer rather than a failure:
single-linkage on real scRNA-seq essentially never separates, which is why the
field clusters k-NN graphs with Leiden instead. The useful output is the
statistic, not the verdict — pbmc is measurably the most patchy, matching
immune types being more discrete than a whole-organism atlas.

## Fiedler value and eigengap

`fiedler` builds a k-NN graph on the embedding with self-tuning affinities
(Zelnik-Manor & Perona: local scale σᵢ = distance to the k-th neighbour, so a
dense cell type and a sparse one are not forced to share a bandwidth), takes
the normalised Laplacian L = I − D^−½WD^−½, and reads its bottom spectrum. No
sparse eigensolver: at these point counts the dense Laplacian is the same size
as the covariance EVD already being run, and a dense symmetric EVD returns the
whole bottom spectrum, so the eigengap comes free next to λ₁.

λ₁ alone is not a classifier, and it fails on exactly the interesting case. A
path graph on N nodes has algebraic connectivity ~(π/N)² while being perfectly
connected: 800 points along a smooth 1-D curve give λ₁ = 1.4e-4, which any
absolute threshold reads as "partitioned". A trajectory is a long thin
manifold, so it will always look weakly connected in absolute terms. The
verdict therefore rests on two things without that failure mode:

- **Exact zeros.** The multiplicity of eigenvalue 0 is the number of connected
  components. (Detected against 1e-8, so "nothing to tune" is an
  overstatement — but the true zeros land at 1e-16 and the next eigenvalue at
  1e-3, so the margin is four orders of magnitude either way.)
- **The relative eigengap** λ_{k+1}/λ_k. For k separated clusters the ratio at
  k is enormous; for a path graph λ_k ~ (kπ/N)², so consecutive ratios are 4,
  2.25, 1.78, … — bounded and shrinking. The default threshold of 5 clears a
  path graph's maximum of 4.

λ₁ is still reported: as algebraic connectivity it genuinely measures how thin
the bottleneck is. It just is not a verdict on its own.

Measured, same three datasets: trachea λ₁ = 5.1e-3 / gap 1.90×, tm-facs
2.5e-3 / 1.64×, pbmc 1.5e-3 / 4.32×. All connected, and pbmc is again the most
patchy — the same ordering β₀ gives from a different construction.

### The ratio does not survive a change of cloud size

Across the same three datasets at ~2000, ~4000 and ~8000 cells the eigengap ran
2.08/2.05/1.93, 3.74/2.93/3.58 and **3.46/4.22/6.40**. On the third it crosses
the 5× threshold somewhere in that range, so the row was flipping from "one
patch" to "5 weakly-joined patches" purely by moving `--geom-cells`. That is
worse than a wrong number: it is a wrong *verdict*, produced by a knob that has
nothing to do with the biology.

The cause is structural. It is a max over `max_patches` order statistics of a
noisy ratio, so it is an extreme-value statistic and grows with whatever it is
maximised over; and the spectrum underneath is itself N-dependent, because a
graph Laplacian's eigenvalues converge to the manifold operator's only after a
bandwidth-dependent rescaling, and without one λ_k → 0 as N grows.

**Renormalising does not rescue it, and this was measured rather than assumed.**
Weyl's law gives λ_k ~ k^(2/d) for a connected d-manifold, so the spacings

```
g_k = (log λ_{k+1} − log λ_k) / (log(k+1) − log k)
```

should be flat at 2/d for any d and any N — and they are *exactly* 2 on a path
graph, which is the elegant part. But the denominator falls from 0.69 at k=1 to
0.025 at k=39, so the transform amplifies high-k spectral noise and manufactures
gaps. Tried on the same three datasets it turned two of them into spurious
patch counts (zheng 5.24 → "5 patches" at one cloud size, 6.19 → "15 patches" at
the next). It was implemented, measured, and reverted; the reasoning is left in
the source so nobody re-derives it.

So the row hedges. A ratio that clears 5× by less than 30% is inside its own
measured drift and gets `MARGINAL` with no patch count. The band is one-sided
and the asymmetry is the point: *below* the threshold there is a principled null
— a path graph tops out at 4 — so anything under 5 is consistent with a
connected manifold however thin, and a symmetric band would swallow that at 4.0
and hedge on the one case this row was designed to get right. Above 5 there is
no scale at all, which is exactly where the drift bites.

Result on the nine runs: every one now reports one patch, eight say `connected`
and only seurat at 8000 cells says `MARGINAL` — where it previously asserted
five. When the row hedges, `betti0` answers the same question from the MST with
no k, no bandwidth and no eigensolver, and its `clumpiness` is the number to
read.

Known gap: a node whose affinities all underflow to zero gets an identity row
and eigenvalue 1, so it is not counted as its own component. Union
symmetrisation makes this rare (every node keeps k edges) but it is possible
for a far outlier next to a dense region.

## Ollivier-Ricci curvature

`betti0` and `fiedler` give global verdicts — is this one piece? `ricci-neg`
gives the local one: which cells sit on the bottlenecks. For an edge (x, y),
put a lazy random-walk measure on each endpoint (mass α at the point, (1−α)/k
over its k neighbours) and compare the cost of moving one to the other against
the distance between them:

```
κ(x, y) = 1 − W₁(m_x, m_y) / d(x, y)
```

Negative κ means the neighbourhoods are harder to align than the endpoints are
far apart — the edge is a bridge. Positive κ means heavy overlap, as inside a
cluster. Aggregated to nodes, the negatively curved cells are branch points of
a trajectory, or the thin joins between cell types that keep `betti0` and
`fiedler` from calling the graph partitioned.

W₁ is solved exactly, by min-cost flow, not by entropic Sinkhorn. Sinkhorn's
penalty biases W₁ upward and therefore κ downward — straight into the quantity
being counted. Each transport problem is only (k+1)×(k+1), so exactness is
cheap: 2000 cells at k=15 takes ~1 s.

### The ground metric decides the sign

| | flat blob (true κ = 0) | bridge vs interior (bridge must be lower) | Setty 2019, κ < 0 |
|---|---|---|---|
| **`geodesic`** (default) | −0.013 ✓ | Δ −0.015 ✓ | 39.0% |
| `hops` | −0.018 ✓ | Δ **+0.103** ✗ wrong sign | 44.4% |
| `euclidean` | **+0.066** ✗ biased | Δ −0.073 ✓ | 0.0% |

`geodesic` — shortest path through the k-NN graph with Euclidean edge weights —
is the only one correct on both controls, and is what the literature uses. It
agrees with straight-line distance for adjacent points (the direct edge is the
shortest path) but grows correctly for points close in ambient space yet far
along the manifold, which is exactly the pair a curvature diagnostic must
price.

`euclidean` lets mass travel along chords the manifold does not contain, so
transport cost is understated and κ comes out positive almost everywhere: on a
branching hematopoiesis trajectory it reported 0 of 1625 cells negatively
curved, where the geodesic finds 633. `hops` throws away the fact that some
k-NN edges are far longer than others and ranks a thin filament as *more*
positively curved than a blob interior.

Three controls with closed-form answers pin the core: K₅ → κ = 3/4, C₈ → κ = 0,
two 2-stars joined at their centres → κ = −2/3 on the bridge.

### Reading the number: `--ricci-cut`

Counting κ < 0 is close to meaningless. Flat space has zero Ricci curvature, so
in a featureless region the distribution straddles zero and about half the
cells land negative on sampling noise alone. An absolute threshold does not
help either — the spread of κ depends on k, on α and on local density, so a cut
that isolates the tail on one dataset sits in the bulk of the next.

`--ricci-cut` is therefore in robust z units of the data's own curvature
distribution (Iglewicz-Hoaglin): z = 0.6745(κ − median)/MAD. Being a ratio of
two quantities in the same units it carries none of its own, so it transfers
across datasets, k and α. 3.5 is the conventional outlier threshold.

The count is paired with the tail asymmetry — cells below −cut against cells
above +cut — which needs no distributional assumption. Symmetric noise about a
flat mean gives ~1; genuine bottlenecks put mass in the left tail with nothing
matching on the right, so the ratio climbs.

| dataset | κ < 0 | beyond −3.5z | beyond +3.5z | asymmetry |
|---|---|---|---|---|
| Setty 2019 (trajectory) | 39% | 0 | 8 | 0.00 |
| tm-droplet-trachea (K=5) | 56% | 0 | 9 | 0.00 |
| pbmc (K=31) | 50% | 0 | 17 | 0.00 |
| **tm-facs (K=81)** | **17%** | **6** | **0** | **6.00** |

The κ < 0 fraction is anti-correlated with the extreme count. Setty is 39%
negative with zero extremes; tm-facs is only 17% negative but is the sole
dataset with a genuine left tail. The sign test was reporting where the median
sits, not whether there is structure. Biologically that reads correctly: an
atlas of 81 distinct types has real seams, a hematopoiesis continuum does not.

Caveat: with 6 extreme cells out of ~1500, the asymmetry column rests on small
counts and no null model. It ranks datasets; it does not test anything.

Worse than that: `low` is zero on six of the seven datasets tried, so the count
ratio reports 0.00 whatever the shape of the left tail. The row's float is
therefore Hinkley's quantile skew, `(q95 + q05 − 2·median)/(q95 − q05)`, which
asks the same question — is there mass on the left with nothing matching on the
right — on percentiles that always exist, bounded in [−1, 1] and free of any
scale. Negative is the direction bottlenecks live in. Measured, it drifts 15–25%
across a 4× cloud against only ~1.5× separation between datasets, so it ranks
weakly; and all three datasets come out *positive*, i.e. right-tailed.

### The single MAD is not the problem — measured

The obvious suspicion about the integer is that Iglewicz-Hoaglin assumes a
symmetric distribution while kappa is bounded above by 1 and should be
left-skewed, so one pooled MAD would be inflated by the left shoulder and mask
the very cells being counted. That was checked directly rather than assumed, and
it is wrong on both halves:

| dataset | median | MAD | MAD left | MAD right | skew | low / high | double-MAD low / high |
|---|---|---|---|---|---|---|---|
| Norman 2019 | −0.0711 | 0.0531 | 0.0499 | 0.0572 | +0.192 | 0 / 11 | 0 / 9 |
| zheng2017 | −0.0201 | 0.0737 | 0.0656 | 0.0833 | +0.186 | 0 / 0 | 0 / 0 |
| seurat pbmc | −0.0116 | 0.0601 | 0.0564 | 0.0623 | +0.121 | 0 / 22 | 1 / 21 |

`MAD left < MAD right` on all three: these distributions are *right*-wider, which
the quantile skew says independently. And splitting the MAD moves the counts by
one cell across three datasets.

So the zero is real. `ricci-neg` reports no bottleneck cells on these three
because there are none — all three are datasets `betti0` and `fiedler` also call
one continuous piece, and one of them is Perturb-seq on a single cell line. The
row does fire where the structure exists: tm-facs, 81 cell types, gives 6 cells
beyond −3.5 with an asymmetry of 6.00 in the table above. Nothing to fix; the
double MAD was implemented as a measurement and not kept.

### Implementation notes

- The min-cost flow uses Dijkstra on reduced costs with potentials, not a
  label-correcting search on raw costs. Two neighbourhoods that share a point
  put a zero-cost cycle in the residual graph, and with distances of order
  10–100 — where f64 spacing is ~1e-14 — no absolute improvement threshold can
  separate "improved" from "rounded", so Bellman-Ford/SPFA relaxes such a cycle
  forever. Non-negative reduced costs remove the failure mode instead of tuning
  around it. The regression test runs the CLI's exact shape at cost scales 1,
  50 and 1e4.
- Approximate nearest neighbours are not needed and not used. Every geometric
  heuristic here is O(m²) or O(m³) and capped at `--geom-cells` ≈ 2000, where
  exact k-NN off the Gram matrix is ~4M distance evaluations — far from the
  bottleneck.

## One cloud, six diagnostics

TwoNN, correlation dimension, Betti-0, Fiedler and Ricci all want pairwise
distances among the same strided subsample, and two of them want the same k-NN
lists. `geom::Cloud` builds the Gram matrix once and hands out `d2(a, b)`;
`Cloud::knn` builds the neighbour lists once, in parallel, sorted so that
Fiedler's local scale sigma_i is just the last entry.

Before this, each of the five built its own Gram — and the TwoNN scale analysis
built a fresh one per decimation level per replicate, about 18 BLAS calls per
run for one matrix. The k-NN scan (an O(m²) `select_nth` per point) was run
twice, once by Fiedler and once by Ricci, both sequentially.

The seven then run concurrently under `rayon::scope`: they share the cloud
read-only and do not talk to each other. Each is internally parallel too, and
nesting is fine — rayon work-steals across the same pool. Measured on Setty
2019 at the default 2000-cell cap, the geometry block goes 1.42 s → 1.09 s,
bounded by Ollivier-Ricci. At `--geom-cells 4000` it is ~12 s → 6.5 s, bounded
by the Laplacian EVD, which is the O(m³) term and will dominate from there on.
Every printed estimate is unchanged.

The per-stage times in the progress output now overlap and do not sum to the
total, which is why they are indented under one heading.

Where the remaining time goes, at the default cap: reading and biwhitening are
~80% of the run, so none of this moves the headline. It buys headroom for a
larger `--geom-cells`, which is the setting that actually limits what the
geometric rows can resolve (the Eckmann-Ruelle ceiling grows as log N).

## Geometry in the signal subspace

TwoNN, correlation dimension, Betti-0, Fiedler and Ricci all run on the PCA
scores truncated at the Tracy-Widom rank, not on the 2000-gene biwhitened
matrix. In 2000 ambient dimensions distances concentrate hard enough to destroy
the measurement: on Tabula Muris FACS the longest MST edge was 1.2× the median
and the largest single-linkage step 1.01×, i.e. 81 well-separated cell types
were indistinguishable from a smooth curve. Projecting first moves those to
2.2× and 1.13×, and TwoNN's estimate from 41 to 13.

The eigenvectors come free with the EVD that runs anyway, so this costs
nothing. The rank comes from a spectral heuristic, so the geometric ones are
not choosing their own embedding.

Two limits of the current wiring: only `SCORE_K = 100` score columns are
retained, so a TW rank above 100 is silently truncated for the geometry (it
happens on both whole-organism atlases, TW 172 and 184); and when TW hits
`--k-max` the embedding is exactly k_max by construction.

## Does a bigger cloud help?

Every geometric row is capped by `--geom-cells`, and every dimension estimator's
resolving power grows with N, so the obvious question is whether the cap is what
is limiting them. Three datasets, `--max-cells 8000` held fixed so the embedding
never changes, clouds of ~2000, ~4000 and ~8000 (the stride is an integer, so
those are the sizes actually reachable — asking for 6000 out of 8000 gives 4000).

| dataset | cloud | twonn | plateau | corr-dim / ceiling | local-pca |
|---|---|---|---|---|---|
| Norman 2019 | 1987 | 21.15 | — | 18.9 / 6.6 | 17.43 @k=256 |
| | 3974 | 22.30 | — | 18.1 / 7.2 | 17.17 @k=512 |
| | 7947 | 23.89 | — | 17.8 / 7.8 | 16.86 @k=1024 |
| zheng2017 | 1972 | 15.73 | — | 13.9 / 6.6 | 10.15 @k=256 |
| | 3944 | 16.19 | 15.43 | 14.5 / 7.2 | 9.79 @k=512 |
| | 7888 | 17.46 | — | 13.8 / 7.8 | 10.40 @k=1024 |
| seurat pbmc | 1926 | 27.48 | — | 19.9 / 6.6 | 22.32 @k=256 |
| | 3852 | 29.40 | 17.56 | 21.2 / 7.2 | 20.38 @k=512 |
| | 7704 | 32.09 | — | 21.3 / 7.8 | 17.50 @k=1024 |

**No, and the reason is different for each row.**

- **`twonn` rises monotonically with the cloud** on all three, 21→24, 16→17,
  27→32. That is the estimator working correctly: more cells means a smaller
  first-neighbour distance, which means more of the full-rank noise inside the
  ball. The raw TwoNN number is a function of the sampling density and should
  never be quoted without one.
- **`corr-dim` is structurally hopeless here.** Four times the cells bought 1.2
  of ceiling, because it grows as 2·log₁₀N. Measuring D ≈ 20 legitimately would
  need ~10¹⁰ cells. The row can only work in the trajectory regime (D 2–5) its
  unit tests cover.
- **`twonn-plateau` flickers.** Plateaus appear at 4000 on two datasets and are
  gone again at 8000. Those were an accident of where the decimation levels
  landed, not a property of the data. Two points looked like a trend; the third
  killed it.
- **`local-pca` was reporting a constant of ours.** With the cap at 256 the dip
  sat on the last rung at every cloud size and the estimate drifted 24–46%
  across the ladder. Raising `K_CAP` to 1024 (with centres falling as 1/k², so
  each doubling costs 2× rather than 8×) makes `k_max = min(1024, m/4)` scale
  with the cloud, and it is the *fraction* of the cloud that turns out to be the
  stable quantity: Norman went 24% → 3% drift, zheng 46% → 6%.

Even at k = 1024 — an eighth of an 8000-cell cloud — the walk is still falling:

```
Norman @8000:  16.0*  26.59  24.96  23.28  21.56  19.61  16.86     (* ceiling-pinned)
zheng  @8000:  16.0*  18.55  17.67  16.60  14.83  12.67  10.40
seurat @8000:  16.0*  32.0*  39.64  35.23  31.29  25.81  17.50
```

So the honest reading of every dimension row on real scRNA-seq at these counts
is **an upper bound**. Two estimators with unrelated failure modes now agree on
that, which is a verdict rather than a number, and it is the thing to say out
loud rather than bury.

Cost: ~7 GB peak and ~80 s at an 8000-cell cloud, dominated by the O(m³)
Laplacian EVD. There is no memory guard on `--geom-cells`.

### Two floats that had to be replaced

The same ladder is the test bed for any statistic that claims to describe the
data rather than the sample.

| statistic | 2000 → 8000 drift | separation between datasets | verdict |
|---|---|---|---|
| MST step ratio (was `betti0`'s) | 11–13%, monotone → 1 | ~3% at 8000 | dead |
| **clumpiness** `log(AM/GM)` | 7–8% | **2×**, same order at every size | keeps |
| ricci tail skew | 15–25% | ~1.5× | weak but honest |
| fiedler eigengap | 3.46 → 6.40 on one dataset | — | **hedged, see below** |

The MST step ratio decays toward 1 as the cloud grows, because a denser MST has
a smaller maximum step whatever the structure — at 8000 cells the three datasets
read 1.04 / 1.06 / 1.07 and are indistinguishable, and at 2000 they are ordered
differently. It was measuring the sample size. `clumpiness` drifts only slightly
less but separates the datasets by 2× with a stable ordering, a 25:1 signal to
drift; and zheng2017, a concatenation of separately sorted PBMC populations and
so the most genuinely discrete of the three, is the one that scores double.

## Measurements on labelled data

Six datasets with ground-truth cell-type labels, all at `--n-genes 2000
--max-cells 5000`:

| dataset | K (labels) | bulk-KS | σ² | TW | MP |
|---------|-----------|---------|-----|----|----|
| tm-droplet-trachea | 5 | 0.031 | 0.74 | 77 | 78 |
| pbmc (raw) | 11 | 0.056 | 0.71 | 80 | 80 |
| tm-facs-marrow | 22 | 0.031 | 0.65 | 84 | 88 |
| pbmc | 31 | 0.048 | 0.74 | 71 | 72 |
| tm-droplet | 55 | 0.095 | 0.57 | 184 | 176 |
| tm-facs | 81 | 0.081 | 0.59 | 172 | 166 |

The TW rank does not track the label count. K goes 5 → 31 while the rank sits
flat at 71–88; the jump to ~170 comes with the two whole-organism atlases, not
with more labels. What it tracks is tissue breadth. That is not a failure of
the estimator — cell-type labels are a coarse discretisation of a signal
subspace that also holds within-type variation, batch, cell cycle and depth —
but "number of clusters" is the wrong yardstick for it, and anyone reaching for
these numbers to pick a PCA rank should know the answer is not K.

bulk-KS is ≤ 0.10 throughout, so the MP assumption holds on all six and the
ranks are valid. σ² lands at 0.57–0.74 rather than 1: biwhitening under-scales,
and that is corrected for.

Caveats on this table:

- All six are at n = 5000. The rank depends on n, so these compare to each
  other and to nothing else.
- The TwoNN column that used to sit here was measured in ambient gene space and
  is superseded by the subspace change above (tm-facs went 41 → 13). It is
  dropped rather than left stale; regenerate it if you want it back.
- `--max-cells` takes cells at a stride, never a prefix. Tabula Muris is
  ordered by tissue, so the first 5000 of its 44779 cells cover 8 of the 81
  types and give quite different answers (TW 130).

## Deferred: scwarp acceleration

Sinkhorn-Knopp is ~70% of the run at 20k cells (12.1 s of 17.1 s) and this path
is dense. `../scwarp` branch `feat/biwhitening` has a sparse CSR Sinkhorn-Knopp
(`scwarp_core::biwhiten::sinkhorn_knopp`, CPU + wgpu + CUDA) plus its own MP
edge/rank/KS diagnostics. Swap to it when that branch reaches `main` — it is
currently 10 commits ahead and 3 behind, so a path dependency does not see it.
Three things measured before switching:

1. **GPU is not the lever.** scwarp's own profile has its dense GPU kernel
   losing to its CPU sparse loop at every size, and the sparse GPU winning only
   ~14% at 20k×2k once one-time setup is counted. Their CUDA entry point runs
   Sinkhorn on the CPU for this reason.
2. **Sparsity is not the lever here.** Top-variance gene selection keeps the
   dense genes — the matrix fed to Sinkhorn is 49.6% dense, where sparse and
   dense cost about the same. It becomes the lever only if whitening moves to
   the full gene set (~7% dense), which is what scwarp argues for anyway.
3. **The iteration count is the lever.** scwarp uses Landa et al.'s plain
   Sinkhorn on the variance matrix and converges in 20–35 iterations; the
   Chardès variant used here oscillates under damping and needs ~225.

Two methodological disagreements to settle before adopting it: scwarp does not
mean-centre after whitening (it argues centring shifts the spectrum off the MP
law this all rests on), and it whitens the whole gene set rather than an HVG
subset.

## Candidates not implemented

- **Parallel analysis / permutation.** Shuffle each gene independently,
  recompute the spectrum, keep components above the permuted λ₁ (or its 95th
  percentile over B permutations). No distributional assumption at all, which
  makes it the right cross-check on whether biwhitening worked — and the right
  way to measure the HVG-selection bias in the TW rank. Costs B full EVDs.
- **Bi-cross-validation** (Owen & Perry 2009). Hold out a submatrix,
  reconstruct it from a rank-k fit of the rest, pick the k minimising held-out
  error. Optimises for prediction rather than for a null hypothesis, and needs
  no noise model.
- **Eigenvalue-gap / elbow.** Largest ratio λ_k/λ_{k+1}, or the knee of the
  scree curve. Cheap, no theory, and worth reporting precisely because it is
  what everyone does by eye.
- **BBP overlap floor.** `rmt_spca::rmt::RmtTheory::predicted_overlap` gives
  the expected cos²θ between an outlier eigenvector and the true direction.
  Instead of "is this above noise?", ask "is this direction estimable?" and cut
  where predicted overlap drops below ~0.5. Different and usually smaller
  answer than TW.
- **Other intrinsic-dimension estimators** (Levina-Bickel MLE, Gride, the
  heterogeneous-ID mixture). Same family as TwoNN, different bias/variance
  trade-offs. Survey: Binnie, Dłotko, Harvey, Malinowski & Yim, *A Survey of
  Dimension Estimation Methods*,
  [arXiv:2507.13887](https://arxiv.org/abs/2507.13887v1).

## Further references

- Denti, Doimo, Laio & Mira, Sci. Rep. **12**, 20005 (2022) — Gride.
- Owen & Perry, Ann. Appl. Stat. **3**, 564 (2009) — bi-cross-validation.
- Landa, Coifman & Kluger, SIAM J. Math. Data Sci. **3**, 388 (2021) —
  biwhitening by the variance matrix.
- Little, Maggioni & Rosasco, Appl. Comput. Harmon. Anal. **43**, 504 (2017) —
  multiscale SVD.
- Iglewicz & Hoaglin, *How to Detect and Handle Outliers*, ASQC (1993) — the
  robust z score.
