//! Rank-selection diagnostics for single-cell dimensionality reduction.
//!
//! Reads a count matrix through `scx-core`, biwhitens it, and reports what
//! several heuristics think the number of signal components is. It picks
//! nothing for you -- when the heuristics disagree, that disagreement is the
//! finding.

pub mod betti;
pub mod corrdim;
pub mod io;
pub mod rank;
pub mod tw;
pub mod twonn;
