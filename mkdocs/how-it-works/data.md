# Data in msmu

## Overview

In LC-MS/MS `proteomics`, data analysis typically follows a **hierarchical path**—starting from evidence-level data (PSM or precursor), progressing to peptides, and finally reaching proteins. Each stage introduces its own set of feature annotations, quantification matrices, and tool-specific metadata. As a result, proteomics data naturally form a **multi-level** and **multi-dimensional** structure (e.g., PSM/precursor, peptide, protein; feature metadata; sample annotations; QC metrics).

To manage these properties consistently, `msmu` adopts `MuData` from the `scverse` ecosystem as the fundamental data format.
`MuData` provides a container that can store multiple `AnnData` objects together, which makes it suitable for proteomics workflows where different processing levels must remain connected and accessible within a single object.

`MuData`, together with its constituent `AnnData` objects, is widely used in scRNA-seq to manage complex data matrices and their associated metadata. The same structure fits proteomics naturally: identification-level attributes, quantification values, and sample information can all be stored cleanly and explored in an integrated way.

`msmu` works with data formatted as a [`MuData`](https://mudata.readthedocs.io/en/latest/) object composed of multiple [`AnnData`](https://anndata.readthedocs.io/en/stable/) modalities.
Therefore, understanding the usage of `MuData` and `AnnData` helps when working with `msmu`.

A `MuData` object used in `msmu` is organized by modalities, each corresponding to a specific processing level such as feature, peptide, and protein:

```python
mdata
```

```python
mdata["feature"]

# or
mdata["protein"]
```

As general AnnData object, each individual modality contains `.var`, `.obs`, `.X`, `uns`, and etc,. 
- A `.var` attribute is filled with features of each level data. For example, in `feature` modality for PSMs, information describing scan number, filename, PEP, q-value, and etc, with `filename.scan` index.
- In `.obs`, metainfo for samples can be stored and initially filenames or TMT channels are used as index.
- `.X` Holds the **quantification** matrix.
- All other unstructured data can be stored in `.uns`.


## Data Ingestion from DB search tools

Although different search tools output either one consolidated table or multiple separate tables, their contents can typically be organized into two main conceptual parts:
- Evidence data
- Quantification data

While each tool’s schema differs, all of them describe the same core identification and quantification features needed to construct peptide- and protein-level data suitable for comparative proteomics.

`read_*` functions in msmu extract the essential columns required for QC and downstream processing and migrate them into the `.var` of the `feature` modality. `read_*` functions are implemented in [`msmu/_read_write/_reader_registry`]()

- `read_*` functions (currently available)
    - `read_sage()`
    - `read_diann()`
    - `read_maxquant()`
    - `read_fragpipe()`

- Inputs
    - `search_dir`: A directory path to search result
    - `label`: used label ("tmt" or "label_free")
    - `aquisition`: aquisition method ("dda", or "dia")
- output
    - `mudata`: Data ingested mudata object

- columns migrated into `mdata["feature"].var`
    - `filename`, `peptide`(modified), `stripped_peptide`, `scan_num`, `proteins`, `missed_cleavages`, `peptide_length`, `charge`, `PEP`, `q-value`

- decoy evidences are isolated from `.var` and stored in `.uns["decoy"]` for later use in FDR calculation.
- Quantification data for **LFQ (DDA)** is stored in `peptide` modality.
- Raw information from a search tool is stored in `mdata["feature"].varm["search_result"]`

### `read_sage()`


### `read_diann()`


### `read_maxquant()`


### `read_fragpipe()`