# 2024 Fulcher et al. Tutorial

This tutorial demonstrates the integrated analysis of single-cell RNA-seq and proteomics data using the `msmu` package, based on the original dataset described in [Fulcher et al. (2024)](https://www.nature.com/articles/s41467-024-54099-z).

> Fulcher, J. M., Markillie, L. M., Mitchell, H. D., Williams, S. M., Engbrecht, K. M., Degnan, D. J., ... & Zhu, Y. (2024). Parallel measurement of transcriptomes and proteomes from same single cells using nanodroplet splitting. Nature Communications, 15(1), 10614.

## Tutorials

- [01 Process scRNAseq Data](01_process_rna_data.ipynb)
- [02 Process Proteomics Data](02_process_protein_data.ipynb)
- [03 Handle Multi-omics Data](03_handle_multi-omics_data.ipynb)

## Environment Setup

`.python-version`: `3.11`

### Dependencies (`pyproject.toml`)

```toml
[project]
name = "2024-fulcher"
requires-python = ">=3.11"
dependencies = [
    "fastprogress==1.0.3",
    "ipykernel>=7.1.0",
    "ipywidgets>=8.1.8",
    "mofax>=0.3.7",
    "msmu>=0.2.6",
    "muon>=0.1.7",
    "pimms-learn>=0.5.0",
    "requests>=2.32.5",
]
```

### Package Versions (`pip list`)

```bash
Package                 Version
----------------------- -----------
anndata                 0.12.7
annotated-types         0.7.0
antlr4-python3-runtime  4.9.3
anyio                   4.12.1
appnope                 0.1.4
apsw                    3.51.2.0
apswutils               0.1.2
array-api-compat        1.13.0
asttokens               3.0.1
beartype                0.22.9
beautifulsoup4          4.14.3
biopython               1.86
blis                    1.3.3
catalogue               2.0.10
category-encoders       2.9.0
certifi                 2026.1.4
charset-normalizer      3.4.4
choreographer           1.2.1
click                   8.3.1
cloudpathlib            0.23.0
cloudpickle             3.1.2
comm                    0.2.3
confection              0.1.5
contourpy               1.3.3
cramjam                 2.11.0
cycler                  0.12.1
cymem                   2.0.13
debugpy                 1.8.19
decorator               5.2.1
donfig                  0.8.1.post1
executing               2.2.1
fastai                  2.8.6
fastcluster             1.3.0
fastcore                1.12.4
fastdownload            0.0.7
fastlite                0.2.4
fastparquet             2025.12.0
fastprogress            1.0.3
fasttransform           0.0.2
filelock                3.20.3
fonttools               4.61.1
fsspec                  2026.1.0
google-crc32c           1.8.0
h11                     0.16.0
h5py                    3.15.1
httpcore                1.0.9
httptools               0.7.1
httpx                   0.28.1
idna                    3.11
iniconfig               2.3.0
inmoose                 0.9.1
ipykernel               7.1.0
ipython                 9.9.0
ipython-pygments-lexers 1.1.1
ipywidgets              8.1.8
itsdangerous            2.2.0
jedi                    0.19.2
jinja2                  3.1.6
joblib                  1.5.3
jupyter-client          8.8.0
jupyter-core            5.9.1
jupyterlab-widgets      3.0.16
kaleido                 1.2.0
kiwisolver              1.4.9
legacy-api-wrap         1.5
llvmlite                0.46.0
logistro                2.0.1
markdown-it-py          4.0.0
markupsafe              3.0.3
matplotlib              3.10.8
matplotlib-inline       0.2.1
mdurl                   0.1.2
mofax                   0.3.7
mpmath                  1.3.0
mrmr-selection          0.2.8
msmu                    0.2.6
mudata                  0.3.2
muon                    0.1.7
murmurhash              1.0.15
narwhals                2.15.0
natsort                 8.4.0
nest-asyncio            1.6.0
networkx                3.6.1
njab                    0.1.1
numba                   0.63.1
numcodecs               0.16.5
numpy                   2.3.5
oauthlib                3.3.1
omegaconf               2.3.0
orjson                  3.11.5
packaging               26.0
pandas                  2.3.3
pandas-flavor           0.8.1
parso                   0.8.5
patsy                   1.0.2
pexpect                 4.9.0
pillow                  12.1.0
pimms-learn             0.5.0
pingouin                0.5.5
pip                     25.3
platformdirs            4.5.1
plotly                  6.5.2
pluggy                  1.6.0
plum-dispatch           2.6.1
polars                  1.37.1
polars-runtime-32       1.37.1
preshed                 3.0.12
prompt-toolkit          3.0.52
protobuf                6.33.4
psutil                  7.2.1
ptyprocess              0.7.0
pure-eval               0.2.3
pydantic                2.12.5
pydantic-core           2.41.5
pygments                2.19.2
pynndescent             0.6.0
pyopenms                3.5.0
pyparsing               3.3.2
pytest                  9.0.2
pytest-timeout          2.4.0
python-dateutil         2.9.0.post0
python-dotenv           1.2.1
python-fasthtml         0.12.39
python-multipart        0.0.21
pytz                    2025.2
pyyaml                  6.0.3
pyzmq                   27.1.0
requests                2.32.5
rich                    14.2.0
scanpy                  1.9.6
scikit-learn            1.8.0
scipy                   1.17.0
seaborn                 0.12.2
session-info            1.0.1
session-info2           0.3
setuptools              80.10.1
simplejson              3.20.2
six                     1.17.0
smart-open              7.5.0
soupsieve               2.8.3
spacy                   3.8.11
spacy-legacy            3.0.12
spacy-loggers           1.0.5
srsly                   2.5.2
stack-data              0.6.3
starlette               0.52.1
statsmodels             0.14.6
stdlib-list             0.12.0
sympy                   1.14.0
tabulate                0.9.0
thinc                   8.3.10
threadpoolctl           3.6.0
torch                   2.9.1
torchvision             0.24.1
tornado                 6.5.4
tqdm                    4.67.1
traitlets               5.14.3
typer-slim              0.21.1
typing-extensions       4.15.0
typing-inspection       0.4.2
tzdata                  2025.3
umap-learn              0.5.11
urllib3                 2.6.3
uvicorn                 0.40.0
uvloop                  0.22.1
wasabi                  1.1.3
watchfiles              1.1.1
wcwidth                 0.3.1
weasel                  0.4.3
websockets              16.0
widgetsnbextension      4.0.15
wrapt                   2.0.1
xarray                  2025.12.0
zarr                    3.1.5
```
