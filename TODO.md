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

- [ ] **Run the ladder on it.** Every other statistic here was validated at
      2000/4000/8000 geometry cells; this one has not been, and the hop bins are
      the obvious thing to drift (a denser graph means more hops to cross the
      same distance).
- [ ] **A permutation null on the top eigenvalue of the mean projector.**
      Currently there is no test: "2× chance" is a rule of thumb. Randomising
      the tangent bases and re-accumulating M gives the null distribution of λ₁
      directly, and it is the difference between "there is a shared direction"
      and "λ₁ is 0.9, as it would be for d/D = 0.48 anyway".
- [ ] **Is the shared direction technical?** The leading shared direction is
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
- [ ] **Name the shared directions.** The overlap is computed in PCA-score
      space, so the shared subspace maps back through the loadings to genes.
      That turns "these regions share 2 directions" into "they share *these two
      programs*", which is the version a biologist can act on.

## Diagnostics not built

Ordered by value per line. All reuse state already computed.

- [ ] **Depth confound.** `corr(score_k, log total_counts)` for the leading PCs.
      `totals` is computed in `io.rs` and thrown away; `scores` already exist.
      ~5 lines. PC1 being library size is the most common real failure in
      scRNA-seq PCA, and biwhitening's row scaling partly removes depth, so a
      *surviving* correlation is strong evidence rather than weak.
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
