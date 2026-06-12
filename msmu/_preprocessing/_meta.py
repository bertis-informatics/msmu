from __future__ import annotations

import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import mudata as md
import pandas as pd
from mudata import MuData

from .._core._provenance import uns_logger
from .._tools import _sdrf_pipelines as sdrf_tools
from ..logging_utils import get_logger

_DATAFRAME_FORMATS = {"dataframe", "df", "generic"}
_TABULAR_TEXT_FORMATS = {"csv", "tsv"}
_PARQUET_FORMATS = {"parquet"}
_SDRF_FORMATS = {"sdrf"}
_INDEX_KEY = "index"

logger = get_logger(__name__)


@dataclass(frozen=True)
class _LoadedMetadata:
    dataframe: pd.DataFrame
    format: Literal["dataframe", "csv", "tsv", "parquet", "sdrf"]
    source: str | Path | None


@uns_logger
def add_meta(
    mdata: MuData,
    metadata: pd.DataFrame | str | PathLike[str],
    *,
    format: Literal["dataframe", "df", "generic", "csv", "tsv", "parquet", "sdrf"] | None = None,
    metadata_on: str | Sequence[str] | None = None,
    match_columns: str | Sequence[str] | None = None,
    obs_columns: str | Sequence[str] | None = None,
    validate_sdrf: bool = True,
    skip_ontology: bool = True,
) -> MuData:
    """
    Attach metadata to MuData observations using one explicit metadata key and one obs key.

    Metadata can be provided as a pandas DataFrame, a local path-like object, or
    a URL-like string. File and URL inputs are read into a pandas DataFrame for
    ``csv``, ``tsv``, ``parquet``, and ``sdrf`` formats. Matching is exact only:
    metadata defaults to its index unless ``metadata_on`` is provided, and obs
    defaults to its index unless ``obs_columns`` is provided.
    """
    if not isinstance(mdata, md.MuData):
        raise TypeError("mdata must be a MuData object.")

    metadata_key = _coalesce_metadata_key(metadata_on=metadata_on, match_columns=match_columns)
    obs_key = _normalise_join_key(obs_columns, argument_name="obs_columns", default=_INDEX_KEY)
    loaded = _load_metadata_input(
        metadata,
        format=_resolve_meta_format(format=format),
        validate_sdrf=validate_sdrf,
        skip_ontology=skip_ontology,
    )
    _validate_metadata_dataframe(loaded.dataframe)

    out = mdata.copy()
    _attach_metadata_to_modalities(
        out,
        loaded.dataframe,
        metadata_key=metadata_key,
        obs_key=obs_key,
    )
    return out


def read_sdrf(
    sdrf_file: str | Path,
    *,
    validate: bool = True,
    skip_ontology: bool = True,
) -> pd.DataFrame:
    """
    Read an SDRF file with pandas while preserving the header text exactly as read.
    """
    metadata = _read_delimited_dataframe(
        sdrf_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    metadata.attrs["sdrf_file"] = str(sdrf_file)
    if validate:
        validate_sdrf_file(metadata, skip_ontology=skip_ontology, source_name=str(sdrf_file))
    return metadata


def validate_sdrf_file(
    sdrf_file: pd.DataFrame | str | Path,
    *,
    skip_ontology: bool = True,
    source_name: str | None = None,
) -> None:
    if isinstance(sdrf_file, pd.DataFrame):
        validation_df = sdrf_file
        source = source_name if source_name is not None else sdrf_file.attrs.get("sdrf_file")
    else:
        validation_df = read_sdrf(sdrf_file, validate=False)
        source = source_name if source_name is not None else sdrf_file

    subject = str(source) if source is not None else "DataFrame"
    logger.info("Validating SDRF metadata for %s.", subject)
    try:
        sdrf_tools.validate_sdrf_dataframe(validation_df, source=source, skip_ontology=skip_ontology)
    except Exception as exc:
        logger.error("SDRF validation failed for %s: %s", subject, exc)
        raise
    logger.info("SDRF validation succeeded for %s.", subject)


def attach_sdrf_metadata(
    mdata: MuData,
    sdrf_file: str | Path | None,
    *,
    validate: bool = True,
    skip_ontology: bool = True,
) -> MuData:
    if sdrf_file is None:
        return mdata
    return add_meta(
        mdata,
        sdrf_file,
        format="sdrf",
        validate_sdrf=validate,
        skip_ontology=skip_ontology,
    )


def merge_sdrf_metadata(
    mdata: MuData,
    sdrf_metadata: pd.DataFrame,
    *,
    sdrf_file: str | Path | None = None,
    metadata_on: str | Sequence[str] | None = None,
    match_columns: str | Sequence[str] | None = None,
    obs_columns: str | Sequence[str] | None = None,
) -> MuData:
    metadata = sdrf_metadata.copy()
    if sdrf_file is not None:
        metadata.attrs["sdrf_file"] = str(sdrf_file)
    return add_meta(
        mdata,
        metadata,
        format="sdrf",
        metadata_on=metadata_on,
        match_columns=match_columns,
        obs_columns=obs_columns,
        validate_sdrf=False,
    )


def _resolve_meta_format(
    *,
    format: str | None,
) -> Literal["dataframe", "df", "generic", "csv", "tsv", "parquet", "sdrf"] | None:
    normalised = None if format is None else str(format).strip().lower()
    if normalised is None:
        return None
    if normalised in _DATAFRAME_FORMATS | _TABULAR_TEXT_FORMATS | _PARQUET_FORMATS | _SDRF_FORMATS:
        return normalised  # type: ignore[return-value]
    raise ValueError("metadata format must be one of 'dataframe', 'df', 'generic', 'csv', 'tsv', 'parquet', or 'sdrf'.")


def _load_metadata_input(
    metadata: pd.DataFrame | str | PathLike[str],
    *,
    format: Literal["dataframe", "df", "generic", "csv", "tsv", "parquet", "sdrf"] | None,
    validate_sdrf: bool,
    skip_ontology: bool,
) -> _LoadedMetadata:
    if isinstance(metadata, pd.DataFrame):
        if format in _TABULAR_TEXT_FORMATS | _PARQUET_FORMATS:
            raise TypeError(f"format='{format}' requires metadata to be a path-like object or URL string.")
        dataframe = metadata.copy()
        source = dataframe.attrs.get("sdrf_file")
        if format == "sdrf":
            if validate_sdrf:
                validate_sdrf_file(
                    dataframe,
                    skip_ontology=skip_ontology,
                    source_name=str(source) if source is not None else "DataFrame",
                )
            return _LoadedMetadata(dataframe, "sdrf", source if isinstance(source, (str, Path)) else None)
        return _LoadedMetadata(dataframe, "dataframe", None)

    input_kind = _detect_metadata_input_kind(metadata)
    source = _normalise_metadata_source(metadata, input_kind=input_kind)
    resolved_format = _resolve_source_metadata_format(source, explicit_format=format)

    if resolved_format == "parquet":
        dataframe = pd.read_parquet(source)
    elif resolved_format == "csv":
        dataframe = _read_delimited_dataframe(source, sep=",")
    elif resolved_format == "tsv":
        dataframe = _read_delimited_dataframe(source, sep="\t")
    else:
        dataframe = read_sdrf(source, validate=validate_sdrf, skip_ontology=skip_ontology)

    return _LoadedMetadata(dataframe, resolved_format, source)


def _detect_metadata_input_kind(metadata: pd.DataFrame | str | PathLike[str]) -> Literal["dataframe", "path", "url"]:
    if isinstance(metadata, pd.DataFrame):
        return "dataframe"
    if isinstance(metadata, PathLike):
        return "path"
    if isinstance(metadata, str):
        return "url" if _is_url_like(metadata) else "path"
    raise TypeError("metadata must be a pandas DataFrame, path-like object, or URL string.")


def _is_url_like(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return bool(parsed.scheme and "://" in value)


def _normalise_metadata_source(source: str | PathLike[str], *, input_kind: Literal["path", "url"]) -> str | Path:
    if input_kind == "url":
        return str(source)
    return Path(source)


def _resolve_source_metadata_format(
    source: str | Path,
    *,
    explicit_format: Literal["dataframe", "df", "generic", "csv", "tsv", "parquet", "sdrf"] | None,
) -> Literal["csv", "tsv", "parquet", "sdrf"]:
    if explicit_format in _DATAFRAME_FORMATS:
        raise TypeError("format='dataframe' requires metadata to be a pandas DataFrame.")
    if explicit_format in _TABULAR_TEXT_FORMATS | _PARQUET_FORMATS | _SDRF_FORMATS:
        return explicit_format  # type: ignore[return-value]

    inferred = _infer_source_metadata_format(source)
    if inferred is None:
        raise ValueError("Could not infer metadata format from source. Pass format='csv', 'tsv', 'parquet', or 'sdrf'.")
    return inferred


def _infer_source_metadata_format(source: str | Path) -> Literal["csv", "tsv", "parquet", "sdrf"] | None:
    source_path = _source_path_for_detection(source)
    lower_source = str(source_path).lower()
    if lower_source.endswith(".sdrf.tsv") or lower_source.endswith(".sdrf.tab") or lower_source.endswith(".sdrf"):
        return "sdrf"

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".tab"}:
        return "tsv"
    if suffix == ".parquet":
        return "parquet"
    return None


def _source_path_for_detection(source: str | Path) -> Path:
    if isinstance(source, Path):
        return source
    if _is_url_like(source):
        return Path(urllib.parse.urlparse(source).path)
    return Path(source)


def _read_delimited_dataframe(source: str | Path, *, sep: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(source, sep=sep, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{source} is empty.") from exc


def _coalesce_metadata_key(
    *,
    metadata_on: str | Sequence[str] | None,
    match_columns: str | Sequence[str] | None,
) -> str:
    if metadata_on is not None and match_columns is not None:
        raise ValueError("Use either metadata_on or match_columns, not both.")
    requested = metadata_on if metadata_on is not None else match_columns
    return _normalise_join_key(requested, argument_name="metadata_on", default=_INDEX_KEY)


def _normalise_join_key(
    key: str | Sequence[str] | None,
    *,
    argument_name: str,
    default: str,
) -> str:
    if key is None:
        return default
    if isinstance(key, str):
        return key
    if not isinstance(key, Sequence):
        raise TypeError(f"{argument_name} must be a string or a sequence of strings.")

    resolved = [str(item) for item in key]
    if not resolved:
        raise ValueError(f"at least one {argument_name} value must be provided.")
    if len(resolved) != 1:
        raise ValueError(f"{argument_name} supports exactly one key in the simplified add_meta contract.")
    return resolved[0]


def _validate_metadata_dataframe(metadata: pd.DataFrame) -> None:
    if metadata.empty:
        raise ValueError("metadata is empty.")
    if metadata.columns.has_duplicates:
        raise ValueError("metadata columns must be unique.")


def _attach_metadata_to_modalities(
    mdata: MuData,
    metadata: pd.DataFrame,
    *,
    metadata_key: str,
    obs_key: str,
) -> None:
    metadata_columns = list(metadata.columns)

    for mod_name in mdata.mod.keys():
        adata = mdata.mod[mod_name]
        aligned = _align_metadata_to_obs(
            adata.obs,
            metadata,
            metadata_key=metadata_key,
            obs_key=obs_key,
            modality=str(mod_name),
        )
        adata.obs = _merge_obs_metadata(adata.obs, aligned)

    global_metadata = _collect_mudata_obs_metadata(mdata, metadata_columns)
    mdata.obs = _merge_obs_metadata(mdata.obs, global_metadata)


def _align_metadata_to_obs(
    obs: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    metadata_key: str,
    obs_key: str,
    modality: str,
) -> pd.DataFrame:
    metadata_values = _metadata_key_values(metadata, metadata_key)
    _validate_unique_metadata_key(metadata_values, metadata_key)

    metadata_lookup = metadata.copy()
    metadata_lookup.index = pd.Index(metadata_values, name=None)

    obs_values = _obs_key_values(obs, obs_key, modality=modality)
    matched_mask = obs_values.notna() & obs_values.isin(metadata_lookup.index)
    if not matched_mask.any():
        raise ValueError(
            f"metadata could not be matched to {modality}.obs using metadata key '{metadata_key}' and obs key '{obs_key}'."
        )

    aligned = metadata_lookup.reindex(obs_values)
    aligned.index = obs.index
    return aligned


def _metadata_key_values(metadata: pd.DataFrame, metadata_key: str) -> pd.Series:
    if metadata_key == _INDEX_KEY:
        return pd.Series(metadata.index, index=metadata.index)
    if metadata_key not in metadata.columns:
        raise ValueError(f"metadata key column not found: {metadata_key}.")
    return metadata[metadata_key]


def _obs_key_values(obs: pd.DataFrame, obs_key: str, *, modality: str) -> pd.Series:
    if obs_key == _INDEX_KEY:
        return pd.Series(obs.index, index=obs.index)
    if obs_key not in obs.columns:
        raise ValueError(f"obs column not found in {modality}.obs: {obs_key}.")
    return obs[obs_key]


def _validate_unique_metadata_key(values: pd.Series, metadata_key: str) -> None:
    duplicate_mask = values.notna() & values.duplicated(keep=False)
    if not duplicate_mask.any():
        return

    duplicates = pd.unique(values[duplicate_mask])
    preview = ", ".join(str(value) for value in duplicates[:5])
    raise ValueError(f"metadata key '{metadata_key}' must be unique. Duplicate values: {preview}.")


def _merge_obs_metadata(obs: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = obs.copy()
    aligned = metadata_df.reindex(merged.index)
    for column in aligned.columns:
        if column in merged.columns:
            merged[column] = merged[column].combine_first(aligned[column])
        else:
            merged[column] = aligned[column]
    return merged


def _collect_mudata_obs_metadata(mdata: MuData, metadata_columns: Sequence[object]) -> pd.DataFrame:
    global_metadata = pd.DataFrame(index=mdata.obs.index)
    for mod_name in mdata.mod.keys():
        mod_obs = mdata.mod[mod_name].obs.reindex(global_metadata.index)
        for column in metadata_columns:
            if column not in mod_obs.columns:
                continue
            if column not in global_metadata.columns:
                global_metadata[column] = mod_obs[column]
            else:
                global_metadata[column] = global_metadata[column].combine_first(mod_obs[column])
    return global_metadata
