from copy import deepcopy
from typing import Any, cast

import anndata as ad
import mudata as md
import pandas as pd

from .._core._provenance import log_provenance
from .._utils._mudata import add_modality as add_modality


# Utility functions for Readers
@log_provenance
def concat(mdatas: dict[str, md.MuData]) -> md.MuData:
    """Concatenate samples, retaining modalities in first-seen order.

    Features and observation annotations use an outer join. Feature metadata
    and uns use the first available value, without filling metadata nulls or
    concatenating lists. Provenance joins both input histories separately.
    At least two flat, sample-aligned (axis=0) MuData inputs are required.
    """
    if len(mdatas) < 2:
        raise ValueError("At least two MuData objects are required.")
    modalities: dict[str, ad.AnnData] = {}
    for name, mdata in mdatas.items():
        if not isinstance(mdata, md.MuData):
            raise TypeError(f"Expected MuData object, got {type(mdata)} for {name}.")
        if mdata.axis != 0:
            raise ValueError("concat requires sample-aligned MuData objects (axis=0).")
        for mod, adata in mdata.mod.items():
            if not isinstance(adata, ad.AnnData):
                raise TypeError(f"Expected AnnData modality, got {type(adata)} for {name}/{mod}.")
            modalities.setdefault(mod, adata)
    if not modalities:
        raise ValueError("At least one modality is required.")

    # Lightweight containers own their masks; concat cannot remove input masks.
    # Empty modalities preserve the union despite md.concat taking an intersection.
    inputs = {}
    for name, mdata in mdatas.items():
        mods = {
            mod: cast(ad.AnnData, mdata.mod[mod])
            if mod in mdata.mod
            else ad.AnnData(
                X=template.X[:0, :0] if template.X is not None else None,
            )
            for mod, template in modalities.items()
        }
        inputs[name] = md.MuData(mods, **_container_annotations(mdata))

    # MuData 0.4.1 incorrectly annotates concat's input as AnnData.
    result = md.concat(
        cast(Any, inputs), join="outer", label="dataset", merge="first", uns_merge="first", pairwise=True
    )
    result = md.MuData(
        {mod: cast(ad.AnnData, result.mod[mod]) for mod in modalities}, **_container_annotations(result)
    )
    # First-value metadata may still reference the inputs; do not copy X/layers.
    for data in [result, *result.mod.values()]:
        data.uns = deepcopy({key: value for key, value in data.uns.items() if key != "_log"})
        for attr in ("varm", "varp"):
            mapping = getattr(data, attr)
            for key in mapping:
                mapping[key] = mapping[key].copy()
    return result


def _container_annotations(mdata: md.MuData) -> dict:
    """Copy containers, not matrices; modality masks are rebuilt by MuData."""
    return {
        "obs": mdata.obs.copy(),
        "var": mdata.var.copy(),
        "uns": {key: value for key, value in mdata.uns.items() if key != "_log"},
        **{
            attr: {
                key: value
                for key, value in getattr(mdata, attr).items()
                if attr not in ("obsm", "varm") or key not in mdata.mod
            }
            for attr in ("obsm", "varm", "obsp", "varp")
        },
    }


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts object- and string-type columns in a DataFrame to categorical.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with object and string columns converted to categorical.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = pd.Categorical(df[col])

    return df
