"""``corr`` must attach its result to the RETURNED MuData.

Regression for a bug where ``corr`` computed the correlation, wrote it to ``obsp['X_corr']`` on a
detached ``.copy()`` of the modality, and then returned the untouched ``mdata`` -- so the result was
silently discarded on every call.
"""

import numpy as np

from msmu._tools._correlation import corr


def test_corr_attaches_result_to_returned_mdata(simple_mdata):
    out = corr(simple_mdata, "psm")

    assert "X_corr" in out["psm"].obsp, "corr() did not attach obsp['X_corr'] to the returned mdata"
    n_obs = out["psm"].n_obs
    assert out["psm"].obsp["X_corr"].shape == (n_obs, n_obs)
    # the correlation was actually computed, not left as an empty / all-NaN placeholder
    assert not np.isnan(out["psm"].obsp["X_corr"]).all()
