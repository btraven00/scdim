//! Loading: stream a count matrix through `scx-core`, keep the top-variance
//! genes, hand back a dense cells x genes matrix.
//!
//! Dense is the point: every rank heuristic here needs a full eigenspectrum,
//! so the matrix has to be small enough for an O(g^3) EVD anyway. Gene
//! selection is what makes that true, not an optimisation.

use anyhow::{bail, Result};
use faer::Mat;
use futures::StreamExt;
use scx_core::{open, OpenOptions};

/// A loaded, gene-selected count matrix.
pub struct Counts {
    /// cells x genes, raw counts (or log-normalised, see [`load`]).
    pub x: Mat<f64>,
    /// Column index of each kept gene in the source file.
    pub genes: Vec<usize>,
    /// Shape of the source matrix, before filtering.
    pub source_shape: (usize, usize),
    /// Nonzeros in `x`. Decides whether a sparse Sinkhorn-Knopp is worth it:
    /// top-variance gene selection keeps the *dense* genes, so the selected
    /// submatrix is far denser than the file it came from.
    pub nnz: usize,
}

/// Stream `path`, keep the `n_genes` most variable genes and at most
/// `max_cells` cells, return them as a dense matrix.
///
/// Cells are taken at a regular stride, never as a prefix: single-cell files
/// are commonly ordered by tissue or plate, so the first N rows are a biased
/// sample. On Tabula Muris FACS the first 5000 of 44779 cells cover 8 of the
/// 81 cell types; every 9th cell covers all 81.
///
/// Variance is measured on log1p(CPM-per-10k) — the usual HVG proxy — but the
/// returned values are raw counts unless `log` is set. Biwhitening is derived
/// for count data, so raw is the default.
pub fn load(path: &str, n_genes: usize, max_cells: usize, log: bool) -> Result<Counts> {
    let (shape, indptr, indices, data) = read_csr(path, max_cells)?;
    let (n, p) = (indptr.len() - 1, shape.1);

    // Per-cell totals, for CPM normalisation.
    let totals: Vec<f64> = (0..n)
        .map(|i| data[indptr[i]..indptr[i + 1]].iter().sum::<f64>())
        .collect();

    // Per-gene sum and sum-of-squares of y = log1p(1e4 * count / total).
    // Zeros contribute 0 to both, so iterating the nonzeros is enough.
    let mut sum = vec![0.0f64; p];
    let mut sumsq = vec![0.0f64; p];
    let mut nz_cells = 0usize;
    for i in 0..n {
        if totals[i] <= 0.0 {
            continue;
        }
        nz_cells += 1;
        let scale = 1e4 / totals[i];
        for k in indptr[i]..indptr[i + 1] {
            let y = (data[k] * scale).ln_1p();
            sum[indices[k] as usize] += y;
            sumsq[indices[k] as usize] += y * y;
        }
    }
    if nz_cells < 2 {
        bail!("{nz_cells} cells have nonzero counts; need at least 2");
    }

    let nf = nz_cells as f64;
    let mut ranked: Vec<usize> = (0..p).collect();
    let var = |j: usize| (sumsq[j] - sum[j] * sum[j] / nf) / (nf - 1.0);
    ranked.sort_unstable_by(|&a, &b| var(b).total_cmp(&var(a)));
    ranked.truncate(n_genes);
    // Genes with no variance carry no signal and break biwhitening (zero column).
    ranked.retain(|&j| var(j) > 0.0);
    if ranked.len() < 2 {
        bail!("fewer than 2 genes with nonzero variance");
    }
    ranked.sort_unstable();

    // Dense fill. `pos[j]` maps a source gene column to its output column.
    let mut pos = vec![usize::MAX; p];
    for (out, &j) in ranked.iter().enumerate() {
        pos[j] = out;
    }
    let mut x = Mat::<f64>::zeros(nz_cells, ranked.len());
    let mut nnz = 0usize;
    let mut row = 0usize;
    for i in 0..n {
        if totals[i] <= 0.0 {
            continue;
        }
        let scale = 1e4 / totals[i];
        for k in indptr[i]..indptr[i + 1] {
            let out = pos[indices[k] as usize];
            if out != usize::MAX {
                let v = if log { (data[k] * scale).ln_1p() } else { data[k] };
                x.write(row, out, v);
                nnz += 1;
            }
        }
        row += 1;
    }

    Ok(Counts {
        x,
        genes: ranked,
        source_shape: (shape.0, p),
        nnz,
    })
}

/// Concatenate the streamed row-chunks into one CSR triple, keeping every
/// `n_obs / max_cells`-th row.
///
/// The stride is known before streaming (the reader reports the shape at open
/// time), so only the kept rows are ever materialised -- memory is bounded by
/// `max_cells`, not by the file. The whole file is still read: skipping ahead
/// would need random access, which the streaming reader does not offer and
/// which would not be faster for a chunked HDF5 dataset anyway.
type Csr = ((usize, usize), Vec<usize>, Vec<u32>, Vec<f64>);
fn read_csr(path: &str, max_cells: usize) -> Result<Csr> {
    futures::executor::block_on(async {
        let mut reader = open(path, &OpenOptions::new(4096)).await?;
        let shape = reader.shape();
        let stride = shape.0.div_ceil(max_cells.max(1)).max(1);
        let mut indptr = vec![0usize];
        let mut indices: Vec<u32> = Vec::new();
        let mut data: Vec<f64> = Vec::new();

        let mut stream = reader.x_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let csr = &chunk.data;
            let vals = csr.data.to_f64_par();
            for r in 0..chunk.nrows {
                if (chunk.row_offset + r) % stride != 0 {
                    continue;
                }
                let (lo, hi) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
                indices.extend_from_slice(&csr.indices[lo..hi]);
                data.extend_from_slice(&vals[lo..hi]);
                indptr.push(indices.len());
            }
        }
        Ok::<_, anyhow::Error>((shape, indptr, indices, data))
    })
}
