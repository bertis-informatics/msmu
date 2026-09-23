# Protein Inference

This page explains how `msmu` infers proteins from peptide-level features through [`msmu.pp.infer_protein`](../reference/pp/infer_protein.md).

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

!!! note "PTM workflows infer proteins on the global dataset only"

    PTM data does not need [`infer_protein`](../reference/pp/infer_protein.md). [`to_ptm`](../reference/pp/to_ptm.md) localises sites from each peptide's own
    accessions and the attached FASTA, and [`adjust_ptm_by_protein`](../reference/pp/adjust_ptm_by_protein.md) resolves a site's denominator by
    translating those accessions through the *global* dataset's `protein_map`, so the global dataset
    is the one to run [`infer_protein`](../reference/pp/infer_protein.md) and [`to_protein`](../reference/pp/to_protein.md) on. Protein groups are a judgement derived
    from one dataset's peptide evidence, so keeping them on the side that produced them means a PTM
    peptide the global run never observed — the normal case under enrichment — is still adjustable
    whenever its protein was quantified there. See [`adjust_ptm_by_protein`](../reference/pp/adjust_ptm_by_protein.md).

## Output

A `MuData` with:

- `mdata["peptide"].var["protein_group"]`: Newly inferred protein group
- `mdata["peptide"].var["peptide_type"]`: Peptide type (`unique` or `shared`).
- Decoys receive the same annotations under `mdata["peptide"].uns["decoy"]`.

Output `MuData` also contains mapping information inside `uns`

- `mdata.uns["peptide_map"]`: peptide → protein group mapping.
- `mdata.uns["protein_map"]`: per-protein mapping with flags for `indistinguishable/subset/subsumable` status.

## Citation

> Nesvizhskii, A. I., & Aebersold, R. (2005). Interpretation of shotgun proteomic data. Molecular & cellular proteomics, 4(10), 1419-1440.
