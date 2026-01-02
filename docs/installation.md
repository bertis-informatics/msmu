# Installation

## Requirements

- Python 3.8 or newer
- (Recommended) A virtual environment such as `venv`, `conda`, `pipenv`, or `uv`

## Install with pip

It is strongly recommended to create virtual environment.

### From Source Distribution

=== "pip"

    ```bash
    curl <_RELEASE_BINARY_URL_>
    pip install msmu-<version>.tar.gz
    python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

=== "pipenv"

    ```bash
    curl <_RELEASE_BINARY_URL_>
    pipenv install msmu-<version>.tar.gz
    pipenv run python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

### From PyPI (planned)

> Once published to PyPI, installation will be:

=== "pip"

    ```bash
    pip install msmu
    python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

=== "pipenv"

    ```bash
    pipenv install msmu
    pipenv run python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

### From Source Repository

If you want the latest version from the repository:

=== "pip"

    ```bash
    git clone https://github.com/bertis-informatics/msmu.git
    cd msmu
    pip install -e .
    python -c "import msmu; print('msmu:', msmu.__version__)"
    ```

=== "pipenv"

    ```bash
    git clone https://github.com/bertis-informatics/msmu.git
    cd msmu
    pipenv install -e .
    pipenv run python -c "import msmu; print('msmu:', msmu.__version__)"
    ```
