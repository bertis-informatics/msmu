# Changelog

All notable changes to `msmu` are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Release versions are derived
from git tags via setuptools-scm.

## [Unreleased]

Changes staged on `dev` for the next merge to `main`. The headline additions are limma
moderated-t differential expression, two protein-level rollup methods, and a guard against
under-powered permutation designs; the merge also brings a large memory-usage reduction in
the search-result readers and several new importers.

> **⚠️ Breaking — `mm.tl.run_de`.** limma is now the **default** DE engine, the permutation test is
> opt-in and always shuffles, and the `measure` / `fdr` parameters are **removed** (the fold-change
> measure and the q-value method are now decided by the engine). See **Changed** and **Removed**
> below before upgrading.

### Added

- **Differential expression — limma moderated-t** (`stat_method="limma"` in `mm.tl.run_de`).
  An empirical-Bayes moderated-t engine that recovers power at small sample sizes where the
  permutation null is degenerate. Supports the two-group main effect, interaction /
  difference-in-differences designs (`interaction`, `interaction_levels`), and covariate
  adjustment (`covariates`). Built on [inmoose](https://inmoose.readthedocs.io/) and validated
  numerically against R limma. (#3)
- **Protein-level rollup methods** — `median_polish` (Tukey's median polish, as used by
  MSstats) and `directlfq` (the DirectLFQ rollup) as `agg_method` options for
  `mm.pp.to_protein` and `mm.pp.to_ptm`. Both model per-peptide response factors and operate
  in log2 space. (#2)
- **Permutation floor warning** — `mm.tl.run_de` now warns when a design is too small for the
  permutation FDR to reach significance (e.g. a 3-vs-3 design floors q at ~0.068 regardless of
  effect size) and points to `stat_method="limma"`. (#4)
- **New readers** — `read_cptac` (CPTAC) and `read_delpi` (DELPI) search outputs.
- **`write_pin`** — export a PSM-level MuData to Percolator input (PIN) format.
- **SDRF support** — an SDRF reader and sample-metadata attachment through `mm.pp.add_meta`.
- **`collapse_obs`** — aggregate observations (e.g. technical replicates) within a MuData.
- **io-level quant importer** and direct `DataFrame` inputs for identification/quantification
  files, with stronger export validation.
- **`drop_search_result`** reader parameter for tighter memory control.
- **Normalisation** gains `batch` and `fraction` key support.
- Support for `ndarray` `obsm` embeddings.

### Changed

- **Memory usage** — readers were unified to a fresh-frame build with in-place mutation and
  early release of unused identification frames, substantially lowering peak memory during
  import across `SearchResultReader` and all subclasses. (#1)
- **SDRF parsing** moved into the preprocessing `add_meta` flow.
- **Plotting** split into thematic modules; volcano plotting routed through a plotting facade.
- **Precursor purity** backend moved to a dedicated `_pyopenms` module.
- Centralized MuData access helpers and extracted shared core helpers.
- Standardized module logging; added ruff linting and pyright to the dev toolchain.
- Removed deprecated anndata/mudata key APIs.
- **Differential expression pipeline** — `mm.tl.run_de` was restructured into four explicit stages
  (data validation → test → fold change → fold-change guidance line), with limma and the
  permutation/simple tests behind a shared engine interface. Each engine expresses its fold change
  in its own central tendency: the permutation/simple tests follow `measure` (default median),
  while limma's is its mean-based model contrast. As a result, for `stat_method="limma"` the
  fold-change guidance line (`fc_pct_*`) and representative values (`repr_ctrl`/`repr_expr`) are now
  mean-based — matching limma's log2FC scale — where they previously followed `measure`. limma's
  per-feature log2FC, moderated-t and p-values are unchanged (still bit-exact to R limma). The
  automatic guidance line is drawn only for plain two-group main effects; it is skipped for limma
  interaction and covariate-adjusted contrasts, whose reported log2FC (a difference-in-differences
  or adjusted coefficient) is not on the scale of the raw label-permutation null. (BID-70)
- **BREAKING — limma is the default DE engine.** `mm.tl.run_de(stat_method=...)` now defaults to
  `"limma"` instead of `"welch"`, because at the small sample sizes typical of proteomics the
  permutation null is degenerate (a 3-vs-3 design floors the q-value near 0.068). `run_de` emits a
  one-time notice when limma runs; pass `stat_method` explicitly to state your choice. limma
  requires an explicit `expr` — for a "vs all other groups" comparison use a permutation method.
  (BID-71)
- **BREAKING — the permutation engines always permute.** `welch` / `student` / `wilcoxon` now always
  run a label-permutation test. `n_resamples` is the number of label shuffles — a **positive
  integer** (default `1000`), no longer accepting `None` — and is not an on/off switch; passing a
  value below 1 raises an error pointing to `stat_method="limma"`. limma ignores it. (BID-71)
- **BREAKING — the fold-change measure follows the test.** The central tendency behind `log2fc`
  (and `repr_ctrl` / `repr_expr`) is now fixed by the statistic so that significance and effect size
  describe the same quantity: `welch`, `student` and `limma` are mean-based, `wilcoxon` is
  median-based. Previously `welch` defaulted to a **median** fold change while testing the **mean**,
  which could disagree in sign. `welch`/`student` fold changes therefore change from median- to
  mean-based. (BID-71)

### Removed

- **BREAKING — `measure` parameter of `mm.tl.run_de`.** The fold-change central tendency is now
  derived from `stat_method` (see Changed). (BID-71)
- **BREAKING — `fdr` parameter of `mm.tl.run_de`.** The q-value method is decided by the engine —
  permutation reports an empirical FDR, limma reports Benjamini-Hochberg. Both `p_value` and
  `q_value` are always returned, so uncorrected significance is available from the `p_value` column.
  (BID-71)
- **BREAKING — the parametric two-sample path** (`simple_test`, previously reached with
  `n_resamples=None`). Parametric significance is now `stat_method="limma"`, which additionally
  moderates the per-feature variance. (BID-71)

### Fixed

- **Empirical FDR `pi0` estimation** summed the permutation null over the wrong axis, which
  collapsed the estimate onto a constant; it now counts per-feature exceedances correctly. (#4)
- Numerous reader correctness fixes: decoy-column handling, required `score`/`PEP`/`purity`
  columns, and file-type / path handling across Sage, DIA-NN, MaxQuant, CPTAC, and DELPI.
- `add_filter` now enforces unique filter names.
- `split_tmt` operates on the `psm` modality.
- **Under-powered DE result** — a design with fewer than two samples per group returned a result
  whose statistic/p/q-values were empty while its fold changes spanned every feature, so `to_df()`
  and `plot_volcano()` raised a length-mismatch error. The statistic/p/q are now NaN at full
  feature length, so the fold-change-only result renders correctly. (BID-70)

[Unreleased]: https://github.com/bertis-informatics/msmu/compare/main...dev
