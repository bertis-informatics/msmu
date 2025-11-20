# msmu

**Python toolkit for LC-MS/<u>MS</u> Proteomics analysis based on <u>Mu</u>Data**

---

## Overview

`msmu` is a Python package for scalable, modular, and reproducible LC-MS/MS `proteomics` data analysis.  
It supports PSM, peptide, and protein-level processing, integrates `MuData` (AnnData) structure, and enables stepwise normalization, batch correction, and statistical testing for biomarker discovery and systems biology.

---

## Key Features

- **Flexible data ingestion** from DIA-NN, Sage (now supporting), and other Database search tools (future)
- **MuData/AnnData-compatible** object structure for multi-level omics
- **Built-in QC**: precursor purity, peptide length, charge, missed cleavage
- **Protein inference**: infer protein with ... rule
- **Normalization options**: log2, quantile, median centering, GIS/IRS
- **Statistical analysis**: permutation-based DE test and FDR
- **PTM support** and stoichiometry adjustment with global dataset
- **Visualization**: PCA, UMAP, volcano plots, heatmaps, QC metrics

---

## File Structure and Input Format

### Accepted inputs

- Sage: folder with PSM tables
- DIA-NN: output folder
- MaxQuant
- FragPipe (MSFragger)
- DelPy

## Output

- Integrated multi-level MuData object (`.h5mu`)
- Summary plots and statistics
- Differentially expressed proteins/sites with FDR

## Roadmap

UPCOMING<br> - ✅ Support DIA-NN and Sage formats<br> - ✅ Normalize and aggregate pipeline<br> - ✅ QC metric visualization<br> - ✅ PTM stoichiometry inference<br>

## Citation

UPCOMING<br>
If you use msmu in your work, please cite:

> Choi and Lee et al., msmu: A Pythonic Framework for Modular Proteomics Analysis, in prep.

## License

UPCOMING<br>
MIT License. See LICENSE for details.

## Quick links

- [빠른 시작](how-to/quickstart.md)
- [API 레퍼런스](reference/)
- [예제 노트북](examples/)
