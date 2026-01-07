# Data in msmu

## Overview

In LC-MS/MS "shotgun" `proteomics`, data analysis typically follows a **hierarchical path**—starting from PSM-level data (PSM or precursor), progressing to peptides, and finally reaching proteins. Each stage introduces its own set of feature annotations, quantification matrices, and tool-specific metadata. As a result, shotgun proteomics data naturally form a **multi-level** and **multi-dimensional** structure (e.g., PSM/precursor, peptide, protein; feature metadata; sample annotations; QC metrics).

To manage these properties consistently, `msmu` adopts [`MuData`](https://mudata.readthedocs.io/en/latest/) from the `scverse` ecosystem as the fundamental data format. [`MuData`](https://mudata.readthedocs.io/en/latest/), together with its constituent [`AnnData`](https://anndata.readthedocs.io/en/stable/) objects, is widely used in scRNA-seq to manage complex data matrices and their associated metadata. The same structure fits proteomics naturally: identification-level attributes, quantification values, and sample information can all be stored cleanly and explored in an integrated way.

`msmu` works with data formatted as a [`MuData`](https://mudata.readthedocs.io/en/latest/) object composed of multiple [`AnnData`](https://anndata.readthedocs.io/en/stable/) modalities.
Therefore, understanding the usage of [`MuData`](https://mudata.readthedocs.io/en/latest/) and [`AnnData`](https://anndata.readthedocs.io/en/stable/) helps when working with `msmu`.

A `MuData` object used in `msmu` is organized by modalities, each corresponding to a specific processing level such as psm, peptide, and protein:

```python
mdata
```

```python
mdata["psm"]

# or
mdata["protein"]
```

As a general AnnData object, each individual modality contains `.var`, `.obs`, `.X`, `uns`, and etc,.

- A `.var` attribute is filled with features of each level data. For example, in `psm` modality for PSMs (or precursors), information describing scan number, filename, PEP, q-value, and etc, with `filename.scan` index.
- In `.obs`, metadata for samples can be stored and initially filenames or TMT channels are used as index.
- `.X` Holds the **quantification** matrix.
- All other unstructured data can be stored in `.uns`.

![](../assets/fig1b.svg){ width="100%" }

## Data Ingestion from DB search tools

Although different search tools output either one consolidated table or multiple separate tables, their contents can typically be organized into two main conceptual parts:

- Identification data - Identification data with feature-level annotations
- Quantification data - Quantitative values for features across samples

While each tool’s schema differs, all of them describe the same core identification and quantification features needed to construct peptide- and protein-level data suitable for comparative proteomics.

`read_*` functions in msmu extract the essential columns required for QC and downstream processing and migrate them into the `.var` of the `feature` modality. `read_*` functions are implemented in `msmu/_read_write/_reader_registry`

- `read_*` functions (currently available)

  - `read_sage()`
  - `read_diann()`
  - `read_maxquant()`
  - `read_fragpipe()`

- Inputs
  - `identification_file`: A file path to identification table
  - `quantification_file`: A file path to quantification table (if applicable) (for tools outputting separate quantification files like Sage)
  - `label`: used label ("tmt" or "label_free")
  - `acquisition`: acquisition method ("dda", or "dia") (for tools supporting both DDA and DIA like MaxQuant)
- output

  - `mudata`: Data ingested MuData object

- columns migrated into `mdata["psm"].var`

  - `filename`, `peptide`(modified), `stripped_peptide`, `scan_num`, `proteins`, `missed_cleavages`, `peptide_length`, `charge`, `PEP`, `q-value`

- decoy features are isolated from `.var` and stored in `.uns["decoy"]` for later use in FDR calculation.
- Quantification data for **LFQ (DDA)** is stored in `peptide` modality.
- Raw information from a search tool is stored in `mdata["psm"].varm["search_result"]`

```python
mdata = mm.read_sage(
    identification_file="path/to/results.sage.tsv",
    quantification_file="path/to/tmt.tsv",
    label="tmt",  # or "label_free"
)

mdata = mm.read_diann(
    identification_file="path/to/report.tsv",
)

mdata = mm.read_maxquant(
    identification_file="path/to/output_file",
    label="tmt",  # or "label_free"
    acquisition="dda",  # or "dia"
)

mdata = mm.read_fragpipe(
    identification_file="path/to/output_file/psm.tsv",
    quantification_file="path/to/quantification_file/combined_modified_peptide.tsv", # for LFQ
    label="tmt",  # or "label_free"
)
```
