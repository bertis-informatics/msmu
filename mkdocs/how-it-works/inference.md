# Protein Inference

This page explains how `msmu` infers proteins from peptide-level features through [`msmu.pp.add_filter`](/reference/pp/add_filter/).

## How protein are inferred

1. **Merge indistinguishable proteins** (`_find_indistinguisable`)  
   Proteins with identical peptide sets are merged and named as a comma-joined list of members.
2. **Collapse subsettable proteins** (`_find_subsettable`)  
   If one protein group’s peptide set is a strict subset of another’s, it is reassigned to the protein group that has larger set.
3. **Handle subsumable proteins** (`_find_subsumable`)  
   Proteins lacking unique peptides are evaluated within connected components of shared peptides. Proteins that cannot be distinguished are merged; components without unique evidence are dropped.

## Input

A `MuData` that has:

- A `peptide` modality containing `var["stripped_peptide"]` and `var["proteins"]` (semicolon-separated accessions per peptide). If decoys exist, they are pulled from `mdata["peptide"].uns["decoy"]`.

## Output

A `MuData` with:

- `mdata["peptide"].var["protein_group"]`: Newly inferenced protein group
- `mdata["peptide"].var["peptide_type"]`: Peptide type (`unique` or `shared`).
- Decoys receive the same annotations under `mdata.uns["decoy"]`.

Output `MuData` also contains mapping information inside `uns`

- `mdata.uns["peptide_map"]`: peptide → protein-group mapping.
- `mdata.uns["protein_map"]`: per-protein mapping with flags for `indistinguishable/subset/subsumable` status.
