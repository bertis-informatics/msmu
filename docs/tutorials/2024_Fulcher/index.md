# 2024 Fulcher et al. tutorial

This tutorial analyzes paired single-cell RNA and proteomics data from C10 and SVEC cells in [Fulcher et al. (2024)](https://www.nature.com/articles/s41467-024-54099-z). msmu processes Sage identifications and FlashLFQ peptide intensities and records the proteomics preprocessing steps. scanpy, PIMMS, muon, and MOFA provide RNA processing, protein imputation, and multi-omics analysis.

> Fulcher, J. M., Markillie, L. M., Mitchell, H. D., Williams, S. M., Engbrecht, K. M., Degnan, D. J., ... & Zhu, Y. (2024). Parallel measurement of transcriptomes and proteomes from same single cells using nanodroplet splitting. Nature Communications, 15(1), 10614.

## Workflow

Use `docs/tutorials/2024_Fulcher` as the working directory. For the main analysis, run **01 → 02** for proteins and **03** for RNA, then combine their outputs in **04**. The protein and RNA notebooks download their inputs. **Appendix A** (workflow export and verification) and **Appendix B** (publication figures) are **optional** and can be skipped when following the main analysis.

| Notebook | Purpose | Main outputs |
| --- | --- | --- |
| [01 Process proteomics data](01_process_protein_data.ipynb) | Filter identifications, normalize peptides, and summarize proteins | `MSV000089280.h5mu` |
| [02 Impute protein data](02_impute_protein_data.ipynb) | Fill missing protein intensities with PIMMS | `MSV000089280_imputed.h5mu` |
| [03 Process scRNA-seq data](03_process_rna_data.ipynb) | Map RNA counts to SDRF cell IDs, filter genes, and normalize | `GSE201575.h5ad` |
| [04 Integrate multi-omics data](04_handle_multi-omics_data.ipynb) | Combine paired modalities, compute WNN UMAP, and fit MOFA | `Fulcher_multimodal.h5mu`, `MOFA_model.hdf5` |
| [Appendix A: Export and verify](A_export_and_validate.ipynb) | Export the recorded workflow and compare protein-processing checkpoints | `requirements.txt`, `script.py` |
| [Appendix B: Generate publication figures](B_figures.ipynb) | Calculate differential expression and checkpoint comparisons, style figure panels, and export SVGs | Figures |

`MSV000089280.h5mu` is the processed checkpoint before imputation, with log₂-transformed, normalized, and summarized protein intensities. `MSV000089280_imputed.h5mu` also contains the PIMMS `imputed` layer used for WNN. Differential expression and MOFA use the observed protein matrix.

The verification section in **Appendix A** and the comparison figures in **Appendix B** require a separately prepared `MSV000089280_debian.h5mu` described in **Appendix A**. **Appendix A** compares the two checkpoints and fits both differential-expression analyses in the current Python session.

## Environment setup

The recorded preprocessing environment uses **Python 3.13.5** and **msmu 0.4.3**. [Download requirements.txt](requirements.txt) for the exported package versions, including scanpy, PIMMS, muon, and MOFA dependencies.
