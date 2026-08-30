# TODO

Reasoning for everything here is in [notes.md](notes.md); this is the checklist.
Anything that adds or changes a reported statistic has to clear **the ladder**
first — see [Does a bigger cloud help?](notes.md#does-a-bigger-cloud-help).

## The acceptance test

Run at ~2000, ~4000 and ~8000 geometry cells with `--max-cells` held fixed so the
embedding never changes, then tabulate two things:

- **drift** of the statistic across the ladder, and
- **separation** between datasets.

Reject anything whose separation does not clearly exceed its drift. `clumpiness`
passes at 25:1 (2× separation, 8% drift); the MST step ratio it replaced had ~3%
separation at 8000 cells against 11–13% drift, i.e. it was reporting the sample
size. Prefer statistics that rest on a whole distribution: every max-of-N order
statistic tried here has failed, because an extremum grows with whatever it is
maximised over.

Two traps that have cost time twice each:

- `cargo test --release` does **not** refresh `target/release/scdim`. Always
  `cargo build --release` before measuring anything.
- `Cloud::new` strides by an integer, so reachable cloud sizes are only n, n/2,
  n/3 … `--geom-cells 6000` out of 8000 rows silently gives 4000.
- `scx` rejects the legacy pre-anndata-0.7 h5ad layout (no `encoding-type`
  attribute) outright. GEO files of that vintage need re-saving through anndata
  before they can be read at all.

## Datasets

- [x] **GSE132188**, pancreatic endocrinogenesis (Bastidas-Ponce 2019) — done,
      see [notes.md](notes.md#the-trajectory-case-gse132188). Every
      dataset measured so far is blob-shaped — Perturb-seq on one cell line, and
      two PBMC sets — and all three read as one continuous piece with no
      bottlenecks. This is the missing case: a real branching trajectory with
      branch points. It is the shape `ricci-neg` was built for and has never
      been tested on here, the D 2–5 regime where `corr-dim` can actually work
      below its ceiling, and the best chance of `local-pca` showing an interior
      dip instead of a monotone fall. Outcome: the curvature tail and `local-pca`
      both separate it cleanly from the blobs; `corr-dim` still does not work;
      the branch points are not resolvable per-cell at 7000 cells.
- [ ] **Raw counts for GSE132188.** The GEO h5ad is log1p-normalised with no
      layers and no `.raw`. Biwhitening is derived for counts and `bulk-KS` came
      back fine (0.051) anyway, but whether the rank moves on the true counts in
      `GSE132188_RAW.tar` is untested.
- [ ] **A second trajectory dataset.** Every conclusion above rests on one.

## Tangent overlap follow-ups

- [x] ~~**Run the ladder on it.**~~ Done. The depth diagnostics pass with 0-4%
      drift and the axis PC is identical at every cloud size; the overlap curve
      fails at 8-36% and is now labelled shape-only. Two fixes were tried and
      both failed -- a distance-calibrated x-axis is worse (34%), and a
      fixed-fraction basis neighbourhood over-corrects (+27% where fixed-count
      is -32%).
- [ ] **A radius-based basis neighbourhood** is the only remaining idea for
      making the overlap curve comparable across runs: fix the physical radius
      rather than the point count, so the tangent estimate is smoothed over the
      same amount of manifold regardless of density. The cost is a variable
      point count per centre, which is bad for subspace estimation in sparse
      regions -- possibly fatal. Only worth trying if the curve is ever needed
      as a number rather than a shape.
- [x] ~~**A permutation null on the mean projector.**~~ Done. Every eigenvalue
      is tested against its own rank's maximum over 20 random-frame ensembles.
      All four datasets have 11-31 shared directions, which overturned the
      guess that zheng and Norman had none. Remaining weakness: the null assumes
      independent frames under H0, while tangent spaces on a connected manifold
      vary smoothly. A stronger null would preserve that smoothness.
- [x] ~~**Is the shared direction technical?**~~ Yes -- the top shared direction
      is library size on all four datasets (|r| = 0.80-0.89), and on seurat PBMC
      it isolates depth better than any leading PC does (0.84 vs 0.58/0.40/0.50).
      Directions 2+ are the biological candidates and are much weaker. Still
      open: whether any of them is cell cycle, which is shared *and* is biology.
      GSE132188 has `proliferation`, `G2M_score`, `S_score` in `obs`.
- [ ] ~~superseded~~ **Is the shared direction technical?** The leading shared direction is
      most likely library size, ambient RNA or cell cycle, since those vary
      inside every region. Testing it is the same computation as the depth
      confound row below: project cells onto the top eigenvector and correlate
      with `log total_counts`. If it is depth, this becomes a label-free
      technical-axis detector, and projecting it out is a batch correction that
      needs no batch labels. Trap: cell cycle is shared *and* is biology, so
      "shared" and "technical" are not the same set. GSE132188 carries
      `proliferation`, `G2M_score` and `S_score`, which makes it the place to
      check before trusting any subtraction.
- [ ] **NMF for naming individual programs.** With 4-7 shared directions the
      subspace has no canonical basis, and non-negativity is the standard way to
      pick one. NMF on the *counts*, not on the tangent spaces -- those are
      signed and NMF cannot touch them. Then push each program's gene vector
      into score space and test its projection onto the leading eigenvectors of
      M: high = shared/activity, low = identity. That is Kotliar's cNMF split
      with a geometric criterion instead of a usage-across-clusters one. Blocked
      on the same prerequisite as everything gene-facing.
- [ ] **The label version.** GSE132188 ships six lineage annotations. One
      tangent space per annotated group gives a pairwise overlap matrix, which
      answers "do the Alpha and Beta branches share directions" directly. Needs
      scdim to read `obs`, which it deliberately never has.
- [ ] **A `--deflate` mode is deliberately not built.** Projecting the shared
      directions out of the scores is one line, and the reasons not to ship it
      are not computational: v1 is only ~69% depth by variance (|r| = 0.83), so
      projection removes the other 31% too; the shared subspace contains cell
      cycle, which is biology; and where v1 tracks a covariate you already
      measured, regressing on the covariate is strictly more targeted than
      projecting out an estimated direction. The geometry's value is *finding*
      an axis, and it earns its keep only where the shared direction correlates
      with nothing you recorded -- an unmeasured ambient or chemistry effect,
      which no regression can reach. A correction mode would also owe users a
      validation loop and a decision about re-fitting the rank afterwards. That
      is a project, not a flag.
- [ ] **Name the shared directions.** The overlap is computed in PCA-score
      space, so the shared subspace maps back through the loadings to genes.
      That turns "these regions share 2 directions" into "they share *these two
      programs*", which is the version a biologist can act on.

## Diagnostics not built

Ordered by value per line. All reuse state already computed.

- [x] ~~**Depth confound.**~~ Done as a side effect of the tangent work --
      `totals` is kept now and the leading PCs' correlation with log depth is
      printed alongside the shared directions. It is only shown under
      `--tangent`; promoting it to a standing row is a few lines.
- [ ] **TwoNN goodness-of-fit.** KS distance between the sorted μᵢ and the
      fitted Pareto(1, d̂), `1 − μ^−d`. Gives TwoNN what `bulk-KS` gives
      Tracy-Widom: a number saying whether the model held. Fails exactly when
      local dimension is heterogeneous or density varies, which the notes
      currently warn about in prose without measuring. ~6 lines, μᵢ already
      sorted.
- [ ] **Eigenvector localisation (IPR).** `1 / Σ vᵢ⁴` per leading eigenvector,
      free from the EVD already run. Separates a population-wide mode from an
      eigenvalue that cleared λ₊ because of a dozen doublets — the check
      Tracy-Widom structurally cannot make, since an outlier eigenvalue is an
      outlier eigenvalue whether it is biology or three bad cells. Two lines.
- [ ] **Effective rank.** `exp(−Σ p log p)` with `pᵢ = λᵢ/Σλ`, or stable rank
      `Σλ/λ₁`. One line, continuous, no threshold. When the integer rows
      disagree, a soft rank is worth having.
- [ ] **Split-half PC reproducibility.** Biwhiten once, split the cells, two
      EVDs, principal angles between the two k-dimensional subspaces; the k
      where cos θ drops below ~0.5 is how many PCs are reproducible. The
      empirical version of the BBP overlap floor, with no noise model at all.
      Two extra EVDs, and the EVD is not the bottleneck — Sinkhorn is ~70%.
- [ ] **Parallel analysis.** Shuffle each gene independently, recompute the
      spectrum, keep components above the permuted λ₁. The only thing that
      *measures* the HVG-selection bias rather than apologising for it, and the
      right cross-check on whether biwhitening worked. Costs B full EVDs.

## Known limits, not yet acted on

- [ ] **`corr-dim` cannot work on atlas-shaped data and should probably say so
      louder.** The Eckmann-Ruelle ceiling grows as 2·log₁₀N: four times the
      cells bought 1.2 of ceiling, and measuring D ≈ 20 legitimately would need
      ~10¹⁰ cells. It is a trajectory-regime estimator (D 2–5, which its unit
      tests cover) being run on data that is nowhere near that. Consider
      declining to print a number above the ceiling rather than printing one and
      shouting.
- [ ] **Every dimension row is an upper bound at these cell counts.** Even at
      k = 1024, an eighth of an 8000-cell cloud, the local-PCA walk has not
      stopped falling. Two estimators with unrelated failure modes agree on
      this. It is stated in the README; there is no fix, only the honesty.
- [ ] **`SCORE_K = 100` silently truncates the geometry** when the
      Tracy-Widom rank exceeds 100, which happens on both whole-organism
      atlases (TW 172 and 184).
- [ ] **A node whose affinities all underflow to zero** gets an identity row in
      the Laplacian and eigenvalue 1, so it is not counted as its own component.
      Rare under union symmetrisation, possible for a far outlier.

## Deferred

- [ ] **scwarp sparse Sinkhorn**, once `feat/biwhitening` reaches `main`. Three
      things measured before switching, and two methodological disagreements to
      settle — see [notes.md](notes.md#deferred-scwarp-acceleration). Sinkhorn is
      ~70% of the run, but neither the GPU nor sparsity is the lever here; the
      iteration count is.
