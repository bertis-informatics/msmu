# ![](./assets/logo.svg){ width="240px" }

**Python toolkit for LC-MS/<u>MS</u> Proteomics analysis based on <u>Mu</u>Data**

## Overview

`msmu` is a Python package for scalable, modular, and reproducible LC-MS/MS bottom-up proteomics data analysis.  
It supports PSM (precursor), peptide, and protein-level processing, integrates `MuData` (`AnnData`) structure, and enables stepwise normalization, batch correction, and statistical testing for biomarker discovery and systems biology.

![](./assets/overview.svg){ width="100%" }

## Key Features

- **Flexible data ingestion** from DIA-NN, Sage and other popular DB search tools
- **MuData/AnnData-compatible** object structure for multi-level omics
- **Built-in QC**: precursor purity, peptide length, charge, missed cleavage
- **Protein inference**: infer protein with parsimony rule
- **Normalization options**: log2 transformation, median, quantile, GIS/IRS
- **Batch correction**: GIS/IRS, median centering
- **Statistical analysis**: permutation-based DE test and FDR
- **PTM support** and stoichiometry adjustment with global dataset
- **Visualization**: PCA, UMAP, volcano plots, heatmaps, QC metrics

## Supporting DB Search Tools

- Sage: [https://sage-docs.vercel.app](https://sage-docs.vercel.app)
- DIA-NN: [https://github.com/vdemichev/DIA-NN](https://github.com/vdemichev/DIA-NN)
- MaxQuant: [https://www.maxquant.org/](https://www.maxquant.org/)
- FragPipe: [https://fragpipe.nesvilab.org/](https://fragpipe.nesvilab.org/)
