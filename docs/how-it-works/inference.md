# Protein Inference

This page explains how `msmu` infers proteins from peptide-level features through [`msmu.pp.infer_protein`](../../reference/pp/infer_protein/).

## How proteins are inferred

Protein inference in `msmu` is performed through a series of incremental refinement steps. By modifying the initial peptide-protein relationship, proteins are grouped based on shared peptide evidence, following principles outlined in Nesvizhskii & Aebersold (2005). The main steps are as follows:

1. **Construct initial peptide-protein graph**  
   A initial graph explaining peptide-protein relationships is constructed.
2. **Merge indistinguishable proteins** (`_find_indistinguishable`)  
   Proteins associated with identical sets of peptides are merged into a single protein group. The protein group is named as a comma-separated list of members.
3. **Collapse subsettable proteins** (`_find_subsettable`)  
   If the peptide set of one protein group is a strict subset of another, protein with smaller peptide set is reassigned to the protein group that has larger peptide set.
4. **Resolve subsumable proteins** (`_find_subsumable`)  
   Proteins lacking unique peptides are evaluated within connected components of shared peptides. Proteins that cannot be distinguished are merged, while components without unique peptide evidence are dropped.
5. **Finalize protein group assignment**  
   After above steps, all remaining protein groups are distinguishable (i.e., having at least one unique peptide). Mappings explaining peptide-protein relationship and annotations describing how each protein was handled are stored in `mdata.uns`.

## Input

A `MuData` that has:

- A `peptide` modality containing `var["stripped_peptide"]` and `var["proteins"]` (semicolon-separated accessions per peptide). If decoys exist, they are pulled from `mdata["peptide"].uns["decoy"]`.

## Usage

Only the `MuData` is required; the reader's own column names are the defaults.

```python
mdata = mm.pp.infer_protein(mdata)
```

Optional arguments:

- `modality` (default `"peptide"`) — modality holding the peptide-level data.
- `protein_colname` (default `"proteins"`) — `.var` column with the semicolon-delimited accessions.
- `peptide_colname` (default `"stripped_peptide"`) — `.var` column with the peptide sequence.
- `propagated_from` (default `None`) — reuse an existing inference instead of running one.

`propagated_from` takes an already-inferred `MuData`, or a path to an `.h5mu` holding one, and
applies its `uns["peptide_map"]` / `uns["protein_map"]` as-is (it raises if the source lacks
them). The usual case is PTM data, which should carry the protein grouping of its matched global
dataset rather than one inferred from modified peptides alone:

```python
ptm_mdata = mm.pp.infer_protein(ptm_mdata, propagated_from=global_mdata)
```

## Output

A `MuData` with:

- `mdata["peptide"].var["protein_group"]`: Newly inferred protein group
- `mdata["peptide"].var["peptide_type"]`: Peptide type (`unique` or `shared`).
- Decoys receive the same annotations under `mdata.uns["decoy"]`.

Output `MuData` also contains mapping information inside `uns`

- `mdata.uns["peptide_map"]`: peptide → protein group mapping.
- `mdata.uns["protein_map"]`: per-protein mapping with flags for `indistinguishable/subset/subsumable` status.

## Citation

> Nesvizhskii, A. I., & Aebersold, R. (2005). Interpretation of shotgun proteomic data. Molecular & cellular proteomics, 4(10), 1419-1440.
