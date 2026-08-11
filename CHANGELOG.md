# Changelog

All notable changes to `msmu` are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Release versions are derived
from git tags via setuptools-scm.

## [0.3.1] - 2026-08-11

### Fixed

- **Protein groups were destroyed when preprocessing a container read back from `.h5mu`** — saving
  stores `var` string columns as categorical, and on pandas 3 `.str.split` over a categorical
  returns the *repr* of the split list, so the following `explode` did nothing:
  `infer_protein` reported `protein_group` as `"['P0C7N9', 'D3Z016']"`, and the PTM path, which
  explodes twice, shredded it further into fragments like `"['A"`. An uninterrupted
  read-to-analysis run was unaffected, but any path through disk was — including
  `infer_protein(propagated_from=...)` and the PTM workflow's global reference container.

### Changed

- **Protein mapping and summarisation group only over values the data contains.** They previously
  opted into `groupby(observed=False)`, which emits a group for every unused category of a
  categorical key — an empty `uns['peptide_map']` entry and a phantom feature row quantified as `0`
  rather than missing. No route to that state through the public API is known, so this is hardening;
  it also matches pandas 3's own default.

## [0.3.0] - 2026-08-11

The headline changes are limma moderated-t differential expression (now the default), model-based
protein rollups, polars-native readers with sparse block-diagonal storage, SDRF-driven sample
metadata, one canonical accession form across readers, and sparse-in / sparse-out preprocessing.

### Added

- **limma moderated-t DE** (`stat_method="limma"`) — an empirical-Bayes engine with power at the
  three-to-four replicates typical of proteomics, where the permutation null is degenerate. Covers
  interaction / difference-in-differences and covariate designs; validated against R limma.
- **`p_adjust`** — selectable multiple-testing correction (R's `p.adjust` family), shared by both DE
  engines; the default `"auto"` keeps each engine's native behaviour.
- **Rollup methods `median_polish` and `directlfq`** for `to_protein` / `to_ptm` — both model
  per-peptide response factors, so every observed peptide contributes instead of a top-N subset.
  Both work in log2 space and drop fully-missing rows first.
- **Sparse block-diagonal `.X`** — DIA-NN precursors and `split_tmt` output keep only observed cells,
  where a dense pivot spends most of its memory on cells that can never be filled.
  `_core/_blockdiag.py` gives every consumer a sparse-aware path so absent cells stay NaN rather than
  reading as measured zeros: `log2_transform`, `normalise`, `scale_data`, `adjust_ptm_by_protein`,
  `correct_batch_effect`, `collapse_obs`, `split_tmt`, the `to_peptide` / `to_protein` / `to_ptm`
  rollups, `run_de`, `corr`, `pca`, `umap`, every `mm.pl.*` plot, and the exporters.
- **SDRF sample metadata** — `attach_sdrf` keeps the validated SDRF immutable in `uns`, where it
  survives `split_tmt` / `collapse_obs`; `apply_sdrf_to_obs` projects key-functional columns onto
  `obs` on demand, matching TMT channels or data files automatically.
- **`contaminant` column from every reader** (previously Sage and MaxQuant only).
- **`collapse_obs`** — aggregate observations such as technical replicates into one sample.
- **`normalise(method="total_sum")`** — constant-sum normalisation for per-sample loading
  differences, previously an unimplemented stub.
- **Grouped normalisation (`group_obs` / `group_var`)** — level plexes, batches or fractions among
  themselves before comparing them.
- **Automatic abundance-scale restore after batch correction** — correction subtracts, so the result
  no longer reads as an abundance; the per-feature scale is restored automatically (IRS for `gis`).
  A fully block-diagonal matrix is deliberately left reference-relative, ready to roll up.
- **`read_delpi`, `read_sdrf`, `write_pin`, io-level `add_quant`**, plus direct `DataFrame` inputs
  and `drop_search_result` for memory control.
- **`ndarray` support for `obsm` embeddings.**

### Changed

- **BREAKING — limma is the default DE engine**, so the default path can return significance at
  proteomics sample sizes. It requires an explicit `expr`; use a permutation method for "vs all
  other groups".
- **BREAKING — the permutation engines always permute.** `n_resamples` is a positive shuffle count
  (default `1000`), not an on/off switch.
- **BREAKING — the fold-change measure follows the test** (`welch` / `student` / `limma` mean-based,
  `wilcoxon` median-based), so effect size and significance describe the same quantity; previously
  `welch` tested the mean but reported a median fold change.
- **BREAKING — `min_pct` defaults to `0.0`** (estimability only). The old `0.5` silently dropped
  low-coverage features, disproportionately the significant ones; raise it from the coverage you
  observe (`pct_ctrl` / `pct_expr`), or pass `0.5` to restore the old default.
- **`run_de` restructured into explicit stages** (validation → test → fold change → fold-change
  guidance line) behind a shared engine interface, so the permutation-null guidance line
  (`fc_pct_*`) is computed independently of the engine and now applies to limma as well. It is drawn
  only for plain two-group main effects, not for interaction or covariate-adjusted contrasts.
- **BREAKING — one canonical accession form, `[rev_][Cont_]<accession>`.** Contaminant detection
  assumed a single spelling of the marker, so recognition depended on which contaminant FASTA the
  search used. Markers (`contam_`, `Cont_`, `CON__`) are now matched position-independently: the
  database field anywhere (decoy tags vary per engine and are a Sage config value, so they are never
  enumerated), the accession field from the start only, and the protein-name field never — so a real
  CONA_CANLI is not misread. A self-duplicated accession stays distinguishable
  (`Cont_P07339;P07339`). `parse_uniprot_accession_group` returns the flag with the accession, so
  readers no longer re-match markers on parsed strings. Removal behaviour is unchanged: the flag is
  recorded, dropping is left to protein-level evidence.
- **BREAKING — MaxQuant and FragPipe emit bare accessions** (`P10000`, not `sp|P10000|X0_HUMAN`),
  matching every other reader so accessions are comparable across engines. MaxQuant still takes its
  contaminant flag from its own "Potential contaminant" column — an explicit engine annotation
  outranks re-deriving one from a string.
- **BREAKING — `normalise(fraction=True)` is now `group_var="filename"`.** The parameters are named
  for what they do (normalise within an obs / var column group) rather than for one instance of it.
  `fraction` and the interim `batch_key` / `fraction_key` remain as warning aliases. `method` is
  also validated at runtime and typed as a `Literal`.
- **Normalisation, scaling and batch correction are sparse-in → sparse-out**, so a sparse container
  no longer reverts to dense at the first preprocessing step. Per-sample methods rescale blocks in
  place; `quantile`, `scale_data` and the NaN-preserving batch methods (`gis`, `median_center`,
  `continuous`) densify to compute but re-sparsify, since they leave the observed-cell pattern
  intact. ComBat stays dense — it replaces the whole array and may fill NaN. Each method's dense and
  sparse behaviour now lives together on the `Normalisation` object.
- **BREAKING — TMT channel labels in `obs` are `TMT126`-style**, so `obs` lines up with SDRF
  `comment[label]` without a translation table. LFQ is unchanged.
- **SDRF-driven `split_tmt`.** `split_tmt(map=None)`, the default, derives the filename → set map
  from the attached SDRF (`comment[data file]` → `set_key`, default
  `comment[sample preparation batch]`; any per-file-constant column may be named, since SDRF has no
  dedicated set column) and records the key in `uns`. `apply_sdrf_to_obs` reads it back and builds
  the `[comment[label], set_key]` composite automatically, so attach → split_tmt → apply needs no
  manual `on`.
- **Readers are polars-native and much lighter** — import peak memory, not analysis, was the limit on
  study size. `.X` is float32 throughout; passing a `DataFrame` still works.
- **BREAKING — public API relocations**, so each function sits where it belongs: `split_tmt` moves to
  `mm.pp` (now on the `psm` modality, emitting a sparse matrix), `add_quant` to `mm.io`. A `_core/`
  package holds shared helpers and plotting is split behind a facade.
- **BREAKING — Python >= 3.12, anndata >= 0.13.0, mudata >= 0.3.10** (the combination that writes
  `.h5mu` correctly on the pandas 3 stack), with new required dependencies polars, directlfq and
  sdrf-pipelines.
- **The plot style is applied per figure** instead of mutating Plotly's global default at import;
  `set_templates()` remains an explicit opt-in.
- **Batch-correction logging is per step** — `gis()` warns which features have no reference and
  become NaN, the restore step reports only what it restored.

### Removed

- **BREAKING — `measure` and `fdr` parameters of `run_de`.** The central tendency follows
  `stat_method` and the q-value method follows the engine; the correction family is now `p_adjust`.
- **BREAKING — the parametric two-sample path** (`n_resamples=None`), superseded by
  `stat_method="limma"`, which moderates the per-feature variance as well.
- **BREAKING — `rescale` parameter of `correct_batch_effect`**, replaced by automatic per-feature
  restore. There is deliberately no replacement knob — restoring against a feature's own level on a
  block-diagonal matrix would undo the correction.
- **BREAKING — `get_modality_dict` and `get_label` exports from `mm.utils`** (internal helpers).

### Fixed

- **Contaminants from the Hao Lab universal FASTA were never flagged** — the Sage reader matched the
  literal `contam_`, missing `sp|Cont_P00722|BGAL_ECOLI`-style entries entirely.
- **Decoys of contaminants lost their marker** (`rev_contam_…` → `rev_P02769`), so targets and their
  decoys could not be removed together — an asymmetric removal biases target-decoy competition.
- **FASTA annotation returned blank on MaxQuant and FragPipe data** — `protein_info` is keyed by
  accession, so unparsed identifiers missed every lookup and Gene / Description / Organism came back
  empty.
- **Filename extension stripping handled only one extension**, leaving `x.mzML.gz` / `x.wiff.scan`
  with a dangling suffix that failed to match SDRF or sample metadata; `add_quant` hard-coded
  `.mzML` and no-op'd on `.raw` / `.d` / `.wiff`. One shared extension list now drives all stripping.
- **`add_quant` destroyed an existing peptide modality** — running `to_peptide` first left it
  discarding `var` and `uns`, so downstream `to_protein` / DE had no identifications. It now fills
  quantification onto the existing axes (and casts to float32, so either ordering converges).
- **Reader correctness** — decoy columns, required `score` / `PEP` / `purity` columns, file-type and
  path handling, Sage `scannr` parsing (a non-`scan=` token now raises instead of yielding a wrong
  scan number), and FragPipe null `Spectrum` values.
- **`.h5mu` writing of reader output** — `varm` names containing `/` (forbidden in an HDF5 key) are
  sanitised and nullable / Arrow-backed string columns are now writable, so saving a freshly read
  container no longer fails.
- **`add_filter` accepted duplicate filter names**, letting one filter silently shadow another.
- **`split_tmt` operated on the `feature` modality instead of `psm`.**

[0.3.1]: https://github.com/bertis-informatics/msmu/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/bertis-informatics/msmu/compare/0.2.10...v0.3.0
