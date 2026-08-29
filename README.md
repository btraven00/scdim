# scdim

Reads a single-cell count matrix and reports what several rank-selection and
geometry heuristics think about it. It does not pick a number for you; where
the heuristics disagree, that is the result.

I/O comes from [`scx`](../scx) (h5ad, h5seurat, 10x h5, mtx, npy). Biwhitening
and Marchenko-Pastur theory come from [`rmt-spca`](../rmt-spca).

Method notes, measurements on labelled data, and the reasoning behind the
defaults are in [notes.md](notes.md).

## Usage

```bash
cargo run --release -- data.h5ad
cargo run --release -- data.h5ad --max-cells 5000 --format json
```

```
Norman_2019.h5ad  3974x2000 of 111255x19018  q=0.5033  density=60.9%  embed=29D  sigma2=0.8709  bulk-KS=0.0103
tracy-widom      29            stopped at component 30: TW1 statistic 2.068, p = 9.276e-3 > 0.001
mp-edge          31            eigenvalues above lambda+ = 2.9221
twonn            18   17.699   d = 17.70 (95% CI 16.93-18.50), 1967/1987 ratios after 1% trim
twonn-plateau    14   13.784   d = 13.78 over N = 62..248 (3 levels within 10%)
corr-dim         15   15.106   D = 15.11 over 7 scales flat to 15% -- AT THE CEILING: D <= 6.6
local-pca        14   14.234   d = 14.23 at the dip (k=256, r=16.79); 16.00 at k=16 -> 14.23 at k=256
betti0            1    0.010   continuous: clumpiness 0.010, largest MST step 1.12x (< 2)
fiedler           1    2.057   connected: lambda_1 = 2.64e-2, largest relative eigengap only 2.06x
ricci-neg         0    0.192   0/1987 beyond -3.5 robust-z; tail skew +0.192; 80% have kappa < 0
```

Followed by four tables: the lowest Laplacian eigenvalues, the correlation
integral, the TwoNN scale analysis, and the local-PCA walk. `--format json`
puts the same numbers plus the raw spectra on one line for `jq`. Progress goes to stderr, so the JSON
stays pipeable; `-q` silences it.

## What the rows mean

**The third column is the continuous reading**, where the row has one. Two rows
have no integer to give: `betti0` has returned 1 and `ricci-neg` 0 on every
dataset ever tried, because "one piece or several" and "how deep are the
bottlenecks" are matters of degree that a count can only answer once a
threshold has already decided them. It is blank for `tracy-widom` and `mp-edge`,
which really are counting eigenvalues.

The middle column is not one quantity. Rows 1-2 are a **rank** (linear
components above the noise floor), rows 3-6 are a **dimension** (of the
manifold the cells lie on), rows 7-8 are a **piece count**, row 9 is a **cell
count**. They are not comparable to each other and none of them is the number
of cell types.

| row | question | number |
|---|---|---|
| `tracy-widom` | how many components beat the Wishart noise floor? | rank |
| `mp-edge` | same, asymptotic limit, no finite-size correction | rank |
| `twonn` | intrinsic dimension from 1st/2nd neighbour ratios | dimension |
| `twonn-plateau` | that estimate at the scale where it stops moving | dimension |
| `corr-dim` | Grassberger-Procaccia slope of log C(r) | dimension |
| `local-pca` | dimension of the local covariance, against neighbourhood size | dimension |
| `betti0` | one connected manifold, or separated patches? | patches (+ clumpiness) |
| `fiedler` | same question via the Laplacian eigengap | patches |
| `ricci-neg` | which cells sit on bottlenecks (extreme negative curvature)? | cells (+ tail skew) |

Rows 3-9 run on the PCA scores truncated at the Tracy-Widom rank, not on the
gene matrix; in 2000 ambient dimensions the distances they need have
concentrated away. The embedding is capped at 100 columns. All seven share one
Gram matrix and one k-NN graph and run concurrently, so their reported times
overlap.

## Read this before trusting the numbers

**bulk-KS** in the header is the load-bearing diagnostic. It is the KS distance
between the noise bulk and the Marchenko-Pastur CDF; `<= 0.10` is a good fit
(Chardès et al. §3.1). Above that the bulk is not MP, and `tracy-widom` and
`mp-edge` mean nothing whatever they print. The tool warns when it happens.
Sinkhorn's own convergence flag is *not* the diagnostic — see notes.md.

**The rank depends on n.** Tracy-Widom power grows with the number of cells, so
`--max-cells` moves the answer. Compare datasets at the same setting.

**Genes are pre-selected by variance**, which is selection on the same
statistic the spectrum measures. Expect a mild upward bias in the rank.

**Every dimension row is an upper bound at these cell counts.** Measured on
three datasets at clouds of 2000, 4000 and 8000 cells: `twonn` rises
monotonically with the cloud (more cells, smaller r, more noise), `corr-dim` is
above its ceiling at every size, and `local-pca` is still falling at the largest
radius the ladder reaches — even at k = 1024, an eighth of the cloud. No scale
reachable here is free of the noise floor. See notes.md for the table.

**`local-pca` reports the dip, not the ends.** The curve is noise-inflated at
small radius and cluster-inflated at large radius; the manifold is the minimum
in between. Both ends are failure modes, and the row says which one it is
sitting on. A radius whose estimate is pinned against `min(k, embed)` is
excluded from the search — pinned levels are pinned *low* and would win it by
construction.

**`corr-dim` has a hard ceiling** of D <= 2·log₁₀ N (Eckmann-Ruelle). Above it
the estimator returns something too small without failing. The row shouts when
it is within 80% of the ceiling; treat those as a lower bound.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-genes` | 2000 | Genes kept, by variance of log1p(CPM). Bounds the O(g³) EVD. |
| `--max-cells` | 20000 | Cells read, at a regular stride (files are often ordered by tissue). |
| `--alpha` | 0.001 | Tracy-Widom significance level. |
| `--k-max` | 100 | Largest rank the sequential test will report. |
| `--log` | off | Log-normalise before biwhitening (biwhitening is derived for counts). |
| `--bw-damp` | 0.3 | Sinkhorn-Knopp under-relaxation. Lower it if biwhitening won't converge. |
| `--bw-max-iter` | 150 | Sinkhorn-Knopp iteration cap. |
| `--geom-cells` | 2000 | Cells in the shared point cloud, used by *every* geometric heuristic (all are O(m²) or worse). Old name `--twonn-cells` still works. |
| `--twonn-trim` | 0.01 | Top fraction of TwoNN distance ratios dropped before fitting. |
| `--twonn-reps` | 3 | Random subsamples per decimation level in the scale analysis. |
| `--knn` | 15 | Neighbours per cell in the Laplacian and Ricci graphs. |
| `--ricci-metric` | geodesic | Ground metric for transport. `hops` and `euclidean` are wrong; see notes.md. |
| `--ricci-cut` | 3.5 | Robust-z cut for "extreme" curvature, in the data's own MAD units. |
| `--format` | txt | `txt` or `json`. |
| `-q` | off | Suppress stage progress on stderr. |

## Pipeline

1. Stream the matrix, keep the top-variance genes, densify.
2. Biwhiten (Sinkhorn-Knopp) so every gene and cell has unit variance. Raw
   counts are heteroskedastic and their bulk spectrum is not MP; nothing below
   works without this step.
3. Mean-centre, form the covariance, take the full eigenspectrum.
4. Estimate σ² by matching the bulk median to the MP median, divide it out.
5. Spectral heuristics on the spectrum; geometric ones on the leading scores.

## Reading the tables

**Correlation integral.** Only the shape of the `slope` column matters. Real
data usually shows three regimes: high and falling at small r (the noise floor,
which is full-rank), a middle minimum or plateau (the manifold, if any), and a
rise at large r. The rise is not saturation — saturation drives the slope to
zero as C → 1. It means that past the within-cluster diameter you start
swallowing whole neighbouring clusters, so C(r) grows faster than a power law:
the set is not self-similar and has no single dimension. Always read the slope
against the ceiling in the `corr-dim` row.

**Local PCA.** Same three regimes as the correlation integral, read off the
covariance instead of the pair counts. `ceiling` is `min(k, embed)`: a k+1-point
neighbourhood spans at most k directions, so a level at its ceiling has measured
nothing. Read `d` against it at every row, and read the dip rather than either
end. Where it agrees with `twonn-plateau` — two estimators with different
failure modes landing on the same number — that is the strongest dimension
evidence the tool produces.

**TwoNN scale analysis.** `d` should fall as N shrinks and then flatten; the
flat part is the answer. Check `<r2>` actually moved — the scale probed goes as
N^(−1/d), so a large d leaves almost no leverage. A 32× decimation moving ⟨r₂⟩
by 1.8× is real scale variation; by 6% is not. `spread` widens at small N, and
a plateau resting only on the noisiest level is not one.

## References

- Chardès et al., *A statistical physics approach to characterise single-cell
  data*, [arXiv:2509.15429](https://arxiv.org/abs/2509.15429) — biwhitening + MP.
- Johnstone, Ann. Statist. 29 (2001); Ma, Bernoulli 18 (2012) — TW₁ centring
  and scaling.
- Patterson, Price & Reich, PLoS Genet. 2 (2006) — the sequential test.
- Facco, d'Errico, Rodriguez & Laio, Sci. Rep. **7**, 12140 (2017) — TwoNN and
  the decimation analysis.
- Denti, `intRinsic`, J. Stat. Softw. (2023) — the reference `twonn_mle`.
- Grassberger & Procaccia, Physica D **9**, 189 (1983); Eckmann & Ruelle,
  Physica D **56**, 185 (1992) — the correlation integral and its ceiling.
- Zelnik-Manor & Perona, NIPS (2004) — self-tuning affinities.
- Little, Maggioni & Rosasco, Appl. Comput. Harmon. Anal. **43**, 504 (2017) —
  multiscale SVD, the local-PCA walk.
- Ollivier, J. Funct. Anal. **256**, 810 (2009) — coarse Ricci curvature.
