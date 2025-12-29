import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData


@pytest.fixture
def sample_mudata() -> MuData:
    """Lightweight MuData object for smoke-testing functions."""
    obs = pd.DataFrame({"group": ["a", "b"]}, index=["cell1", "cell2"])
    var = pd.DataFrame(index=["feat1", "feat2", "feat3"])
    X = np.arange(6).reshape(2, 3)

    adata = AnnData(X=X, obs=obs, var=var)
    return MuData({"rna": adata.copy(), "protein": adata.copy()})
