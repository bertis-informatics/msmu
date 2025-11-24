# msmu

**Python toolkit for LC-MS/<u>MS</u> Proteomics analysis based on <u>Mu</u>Data**

---

## Overview

`msmu` is a Python package for scalable, modular, and reproducible LC-MS/MS `proteomics` data analysis.  
It supports evidence (PSM, or precursor), peptide, and protein-level processing, integrates `MuData` (AnnData) structure, and enables stepwise normalization, batch correction, and statistical testing for biomarker discovery and systems biology.

---

## Key Features

- **Flexible data ingestion** from DIA-NN, Sage, and other Database search tools
- **MuData/AnnData-compatible** object structure for multi-level omics
- **Built-in QC**: precursor purity, peptide length, charge, missed cleavage, etc,
- **Protein-group inference**: infer protein groups under palsimony rule
- **Normalization options**: log2, quantile, median centering, GIS/IRS
- **Statistical analysis**: permutation-based DE test and FDR
- **PTM support** and stoichiometry adjustment with global dataset
- **Visualization**: PCA, UMAP, volcano plots, heatmaps, QC metrics

---

## Supporting DB Search Tools

- DelPy
- DIA-NN
- FragPipe (MSFragger)
- MaxQuant
- Sage


## Citation

UPCOMING<br>
If you use msmu in your work, please cite:

> Choi and Lee et al., msmu: A Pythonic Framework for Modular Proteomics Analysis (in prep).

## License

UPCOMING<br>
MIT License. See LICENSE for details.
