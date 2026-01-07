# Protein Inference

This page explains how `msmu` infers proteins from peptide-level features through [`msmu.pp.add_filter`](/reference/pp/add_filter/).

## How proteins are inferred

1. **Classify peptides**  
   Peptides are classified as `unique` or `shared` based on their mapping to proteins.
2. **Merge indistinguishable proteins** (`_find_indistinguishable`)  
   Proteins with identical peptide sets are merged and named as a comma-joined list of members.
3. **Collapse subsettable proteins** (`_find_subsettable`)  
   If one protein group’s peptide set is a strict subset of another’s, it is reassigned to the protein group that has larger set.
4. **Handle subsumable proteins** (`_find_subsumable`)  
   Proteins lacking unique peptides are evaluated within connected components of shared peptides. Proteins that cannot be distinguished are merged; components without unique evidence are dropped.
5. **Mapping distinguishable proteins**  
   After processing, each protein group is distinguishable. A mapping of original proteins to their final inferred groups is stored.

## Input

A `MuData` that has:

- A `peptide` modality containing `var["stripped_peptide"]` and `var["proteins"]` (semicolon-separated accessions per peptide). If decoys exist, they are pulled from `mdata["peptide"].uns["decoy"]`.

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
