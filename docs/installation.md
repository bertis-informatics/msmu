# Installation

## Requirements

- Python 3.12 or newer
- A separate virtual environment for your analysis

Runtime dependencies are installed automatically, including `anndata` / `mudata`,
`polars`, `scipy`, `scikit-learn`, `statsmodels`, `inmoose`, `directlfq`,
`sdrf-pipelines`, `pyopenms`, and `plotly`.

## Install from PyPI

Create and activate an environment, then install the released package:

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install msmu
    python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

=== "Windows PowerShell"

    ```powershell
    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install msmu
    python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

Use a Python interpreter meeting the requirement above. For a reproducible analysis,
record the printed version and pin that version in your environment specification.
The [home page](index.md) shows the version and source revision used to build these docs;
the published site follows release tags, while repository documentation follows its checkout.
Development-only APIs can be ahead of the PyPI release.

Next, follow the [Quick Start](tutorials/quick_start.ipynb). To run the notebook
interactively in this environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

Download the notebook using its source link on the tutorial page and run its cells
in order. The Quick Start fetches a small, revision-pinned dataset and writes results
under `msmu_quick_start_output` in the notebook's working directory.

## Development checkout

To use the APIs described in the `dev` branch, install that branch explicitly.
Activate a Python >=3.12 virtual environment first, then run:

```bash
git clone --branch dev https://github.com/bertis-informatics/msmu.git
cd msmu
python -m pip install -e .
python -c "import msmu; print('msmu:', msmu.__version__)"
git rev-parse HEAD
```

An editable installation uses this checkout's source. Record its commit as well as
the package version; changing the checkout changes the installed code.
To reproduce another revision, check out that commit or release tag before installing.

## Build and check documentation

From the repository root, in the development environment:

```bash
python -m pip install -e ".[dev,docs]"
python -m mkdocs build --strict
python -m pytest --nbval-lax docs/tutorials/quick_start.ipynb
```

The notebook includes checks for nonempty results, sample metadata, a generated figure,
and an `.h5mu` round trip. It needs network access to fetch its pinned input files.
Both checks also run in PR CI. The site build itself renders saved notebook outputs
without executing notebooks, so a successful build alone does not validate analysis code.
Use `python -m mkdocs serve` to inspect navigation, examples, API pages, and figures locally.

## Plot display and image export

The Quick Start uses interactive Plotly figures and saves a self-contained HTML figure.
If a notebook does not display a figure, open the saved HTML file in a browser.
Some longer tutorials use the PNG renderer. PNG/SVG/PDF export through Kaleido requires
a compatible Chrome installation; if needed, run `plotly_get_chrome` in your environment.
Switching to an interactive renderer avoids that requirement for notebook display.

## Historical tutorials

The [Fulcher case study](tutorials/2024_Fulcher/index.md) documents a separate,
older Python 3.11 / `msmu` 0.2.6 environment with additional multi-omics dependencies.
Follow its versioned environment for historical reproduction, not for current `msmu` APIs.
