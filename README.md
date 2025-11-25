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

## Installation
We recommend using `pipenv` to set up and manage the project environment.

### (optional) pipenv:
```bash
pip install pipenv
```

### msmu Installation:  
```bash
# make pipenv environment directory 
# (any name for environment possible; in here, msmu)
mkdir msmu && cd msmu

# clone msmu git
git clone git@bitbucket.org:bertis/msmu.git
```

``` bash
# install msmu to your environment
pipenv install msmu
```

## File Structure and Input Format
### Accepted inputs
- Sage: folder with PSM tables
- DIA-NN: output folder

## Documentation
UPCOMING

## Citation
UPCOMING<br>
If you use msmu in your work, please cite:
>Choi and Lee et al., msmu: A Pythonic Framework for Modular Proteomics Analysis, in prep.

## License
UPCOMING<br>
MIT License. See LICENSE for details.