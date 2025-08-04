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

## Quick Start
### Load msmu
``` python
import msmu as mm
```

### Load DB search result
``` python
# Load TMT Sage output
mdata = mm.read_sage(sage_output_dir=search_dir, label="tmt")

# Load LFQ Sage output
mdata = mm.read_sage(sage_output_dir=search_dir, label="lfq")

# Load DIA-NN output
mdata = mm.read_diann(sage_output_dir=search_dir)

# save mdata as h5mu (optional)
mdata.write_h5mu("loaded.h5mu")
```

``` python
mdata
```

result:
```
MuData object with n_obs × n_vars = 6 × 236095
uns:	'protein_info'
1 modality
    feature:	6 x 236095
    var:	'proteins', 'peptide', 'filename', 'scan_num', 'charge', 'missed_cleavages', 'semi_enzymatic', 'spectrum_q', 'peptide_q', 'protein_q', 'stripped_peptide', 'modifications', 'observed_mz'
    uns:	'level', 'search_engine', 'quantification', 'label', 'search_output_dir', 'search_config'
    varm:	'search_result'
```

### Data Processing
<details><summary>TMT (DDA)</summary>



#### design_df example:
|tag|sample|condition|
|---|------|---------|
|126|sample_1|control|
|127N|sample_2|treatment|


``` python
# load mdata from h5mu (optional: if h5mu saved)
mdata = mm.read_h5mu("loaded.h5mu")

# rename samples 
# tmt_1 -> sample_1
mdata = mm.rename_obs(mdata, map=design_df[["tag", "sample"]])

# and add meta(design_df) to mdata.obs
mdata.obs = mdata.obs.join(design_df, how="left")
mdata.push_obs()

# filter PSMs with q-values, precurosr isolation purity, decoy/contam, all-Nan
# this only mark PSMs to be filtered
mdata = mm.pp.add_prefix_filter(mdata, prefix=("rev_", "contam_"))
mdata = mm.pp.add_q_value_filter(mdata, threshold=0.01)
mdata = mm.pp.add_all_nan_filter(mdata, modality="feature")

# calculate Precursor isolation purity and mark low purity PSMs
mdata = mm.pp.add_precursor_purity_filter(
    mdata, threshold=0.7, mzml_files=mzml_files, n_cores=10
    )

# filter
mdata = mm.pp.apply_filter(mdata=mdata, modality="feature")

# drop blank sample columns
mdata = mdata[mdata.obs_names.str.contains("BLANK") == False].copy()

# protein inference
mdata = mm.pp.infer_protein(mdata=mdata)

# data processing
## log2 transform
mdata = mm.pp.log2_transform(mdata=mdata, modality="feature")

## normalsation across samples
mdata = mm.pp.normalise(
    mdata=mdata, 
    modality="feature", 
    method="quantile", # quantile, median
    fraction=True # for fractionated TMT data, set "fraction=True" to normalise within each fraction
    )

## batch correction
mdata = mm.pp.feature_scale(
    mdata=mdata, 
    modality="feature",
    method="gis", # median_center if no GIS
    gis_prefix="POOLED",
    rescale=True
    )

mdata = mm.pp.to_peptide(mdata=mdata)
mdata = mm.pp.to_protein(
    mdata=mdata, 
    top_n=3,
    rank_method="max_intensity"
    )
```
</details>

<details><summary>LFQ (DDA)</summary>

### LFQ with Sage (DDA):

``` python
# Load Sage output
mdata = mm.read_sage(sage_output_dir=search_dir, label="lfq")
```

</details>

<details><summary>DIA</summary>

### DIA with DIA-NN:
``` python

```

</details>

### DE analyis
``` python
res = mm.tl.dea.permutation_test(
                mdata=mdata,
                modality="protein",
                control=ctrl,
                expr=expr,
                category="condition", # grouping columns in .obs
                statistic="welch",
                n_resamples=1000,
                n_jobs=1,
            )
```

### Plots
``` python

```

## File Structure and Input Format
### Accepted inputs
- Sage: folder with PSM tables
- DIA-NN: output folder

## Output
- Integrated multi-level MuData object (`.h5mu`)
- Summary plots and statistics
- Differentially expressed proteins/sites with FDR

## Documentation
UPCOMING

## Roadmap
UPCOMING
- ✅ Support DIA-NN and Sage formats
- ✅ Normalize and aggregate pipeline
- ✅ QC metric visualization
- ✅ PTM stoichiometry inference
- 🔲 GUI via Streamlit or Jupyter widget
- 🔲 Integration with LIMS / AWS Batch

## Citation
UPCOMING<br>
If you use msmu in your work, please cite:
>Choi and Lee et al., msmu: A Pythonic Framework for Modular Proteomics Analysis, in prep.

## License
UPCOMING<br>
MIT License. See LICENSE for details.