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


@uns_logger
def attach_sdrf(
    mdata: MuData,
    sdrf: pd.DataFrame | str | PathLike[str],
    *,
    validate: bool = True,
    skip_ontology: bool = True,
) -> MuData:
    """
    Attach an SDRF table to ``mdata.uns["sdrf"]`` as the immutable source of truth.

    The SDRF is stored whole because its rows span both axes -- ``comment[label]`` maps
    to the obs axis (channels) and ``comment[data file]`` to the var axis (runs/fractions) --
    so it is not reducible to obs alone. ``uns`` is copied through ``split_tmt`` and
    ``collapse_obs``, making it the durable home for the original table; obs is left
    untouched here. Use :func:`apply_sdrf_to_obs` to project selected columns onto obs.

    Parameters:
        mdata: MuData to annotate.
        sdrf: An SDRF as a pandas DataFrame, or a path/URL read via :func:`read_sdrf`.
        validate: Validate the SDRF with ``sdrf-pipelines`` (if installed). Default True.
        skip_ontology: Skip ontology term checks during validation. Default True.

    Returns:
        A copy of ``mdata`` with the SDRF DataFrame stored at ``uns["sdrf"]``.
    """
    if not isinstance(mdata, md.MuData):
        raise TypeError("mdata must be a MuData object.")

    if isinstance(sdrf, pd.DataFrame):
        sdrf_frame = sdrf.copy()
        if validate:
            source = sdrf_frame.attrs.get("sdrf_file")
            validate_sdrf_file(
                sdrf_frame,
                skip_ontology=skip_ontology,
                source_name=str(source) if source is not None else "DataFrame",
            )
    else:
        sdrf_frame = read_sdrf(sdrf, validate=validate, skip_ontology=skip_ontology)

    out = mdata.copy()
    out.uns["sdrf"] = sdrf_frame
    return out


@uns_logger
def apply_sdrf_to_obs(
    mdata: MuData,
    *,
    on: str | Sequence[str] | None = None,
    columns: str | Sequence[str] | None = None,
    set_index: str | None = None,
) -> MuData:
    """
    Project columns of the attached SDRF (``uns["sdrf"]``) onto each modality's obs.

    The SDRF spans both axes, so it is reduced to the obs axis by the match key ``on`` and
    only columns that are a *function* of that key (one value per key group) are projected.
    Columns that vary within a key (e.g. ``comment[fraction identifier]`` under a channel key)
    cannot be represented on the obs axis; they are left untouched in ``uns["sdrf"]`` and
    skipped with a warning -- or, if named explicitly in ``columns``, raise. obs never
    silently collapses SDRF data.

    Parameters:
        mdata: MuData with an SDRF attached via :func:`attach_sdrf`.
        on: SDRF column (or list of columns) matched against ``obs.index``. Default None auto-picks:
            after ``split_tmt`` (which records its ``set_key`` in ``uns``) it builds the composite
            ``[comment[label], set_key]`` matching the ``channel_set`` obs automatically; otherwise
            ``comment[label]`` for TMT and ``comment[data file]`` elsewhere. Pass a str or list to
            override (a list forces a "_"-joined composite key).
        columns: SDRF column(s) to project. Default None projects every projectable column;
            naming a non-projectable column raises instead of skipping it.
        set_index: After projection, replace ``obs.index`` with this SDRF column's values.
            Must be projectable and unique across obs. Default None keeps the index.

    Returns:
        A copy of ``mdata`` with projected SDRF columns merged into each modality's obs;
        ``uns["sdrf"]`` is left unchanged as the source of truth.
    """
    if not isinstance(mdata, md.MuData):
        raise TypeError("mdata must be a MuData object.")
    if "sdrf" not in mdata.uns:
        raise ValueError("No SDRF attached to mdata.uns['sdrf']; call attach_sdrf first.")

    sdrf = mdata.uns["sdrf"]
    if not isinstance(sdrf, pd.DataFrame):
        raise TypeError("mdata.uns['sdrf'] must be a pandas DataFrame (attach via attach_sdrf).")

    # File-name matching must ignore extensions: readers store bare stems (e.g. "QExHF03751")
    # while SDRF comment[data file] carries ".mzML"/".raw"/".d". Normalise to stems for matching and
    # projection (uns['sdrf'] keeps the originals), mirroring split_tmt's filename handling.
    if "comment[data file]" in sdrf.columns:
        sdrf = sdrf.copy()
        sdrf["comment[data file]"] = sdrf["comment[data file]"].astype(str).str.rsplit(".", n=1).str[0]

    if on is None:
        split_set_key = mdata.uns.get("tmt_split_set_key")
        if split_set_key is not None:
            # obs are channel_set after split_tmt; match on the (label, set) composite automatically
            on = ["comment[label]", split_set_key]
    on, sdrf, composite_parts = _resolve_match_on(on, sdrf)
    requested_columns = _normalise_sdrf_columns(columns)

    out = mdata.copy()
    projected_columns: list[str] = []  # preserve SDRF column order, deduped across modalities
    for mod_name in out.mod.keys():
        adata = out.mod[mod_name]
        match_key = on if on is not None else _default_sdrf_match_key(adata)
        for column in _project_sdrf_onto_obs(
            adata.obs,
            sdrf,
            match_key=match_key,
            requested_columns=requested_columns,
            set_index=set_index,
            modality=str(mod_name),
            exclude=composite_parts,
        ):
            if column not in projected_columns:
                projected_columns.append(column)
    out.update()
    # update() syncs obs *names* across modalities but not obs *columns*, so merge the projected
    # columns onto the MuData-level obs too (consumers like correct_batch_effect read mdata.obs).
    if projected_columns:
        out.obs = _merge_obs_metadata(out.obs, _collect_mudata_obs_metadata(out, projected_columns))
    else:
        # Report the fact, not a verdict: an empty projection can be multi-set TMT (needs split_tmt),
        # a mismatched `on`/SDRF, or simply that obs and SDRF don't share a granularity. Some obs
        # staying unmatched is normal too (blank/reference TMT channels absent from the SDRF), so this
        # is a hint to inspect -- not an instruction.
        logger.warning(
            "apply_sdrf_to_obs: nothing was projected onto obs. Possible causes: multi-set TMT "
            "(obs are channels but the SDRF spans channel x set -> split_tmt first), or the `on` key "
            "/ SDRF columns not lining up. Inspect obs vs uns['sdrf'] and decide."
        )
    return out


def _default_sdrf_match_key(adata) -> str:
    """TMT obs are channels (``comment[label]``); label-free/DIA obs are files (``comment[data file]``)."""
    return "comment[label]" if adata.uns.get("label") == "tmt" else "comment[data file]"


def _resolve_match_on(
    on: str | Sequence[str] | None, sdrf: pd.DataFrame
) -> tuple[str | None, pd.DataFrame, list[str]]:
    """Resolve the match spec. A list becomes a synthetic "_"-joined key column so a post-split
    ``channel_set`` obs index (e.g. "TMT126_set1") lines up with SDRF ``(comment[label], batch)``.

    Returns ``(resolved_key, sdrf, composite_parts)``; the composite parts are excluded from
    projection since ``obs.index`` already encodes them.
    """
    if on is None or isinstance(on, str):
        return on, sdrf, []
    keys = [str(key) for key in on]
    if not keys:
        raise ValueError("on must name at least one SDRF column when given as a list.")
    missing = [key for key in keys if key not in sdrf.columns]
    if missing:
        raise ValueError(f"SDRF lacks composite match column(s): {missing}.")
    composite_name = "+".join(keys)
    sdrf = sdrf.copy()
    sdrf[composite_name] = sdrf[keys].astype(str).agg("_".join, axis=1)
    return composite_name, sdrf, keys


def _normalise_sdrf_columns(columns: str | Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    resolved = [str(column) for column in columns]
    if not resolved:
        raise ValueError("columns must name at least one SDRF column when provided.")
    return resolved


def _project_sdrf_onto_obs(
    obs: pd.DataFrame,
    sdrf: pd.DataFrame,
    *,
    match_key: str,
    requested_columns: list[str] | None,
    set_index: str | None,
    modality: str,
    exclude: Sequence[str] = (),
) -> list[str]:
    if match_key not in sdrf.columns:
        raise ValueError(f"SDRF match key '{match_key}' not found in SDRF columns (modality '{modality}').")

    if requested_columns is None:
        skip = {match_key, *exclude}
        candidate_columns = [column for column in sdrf.columns if column not in skip]
    else:
        candidate_columns = requested_columns
    projectable, skipped = _resolve_projectable_columns(sdrf, match_key, candidate_columns)

    if skipped:
        if requested_columns is not None:
            raise ValueError(
                f"SDRF columns are not a function of '{match_key}' (they vary within a key): {skipped}."
            )
        logger.warning(
            "apply_sdrf_to_obs: %s columns not projectable under '%s'; kept only in uns['sdrf']: %s",
            modality,
            match_key,
            skipped,
        )

    obs_keys = pd.Index(obs.index)
    sdrf_keys = set(sdrf[match_key])
    unmatched = [key for key in obs_keys if key not in sdrf_keys]
    if unmatched:
        logger.warning(
            "apply_sdrf_to_obs: %s has %d obs key(s) absent from SDRF['%s']; left unmatched (NaN): %s",
            modality,
            len(unmatched),
            match_key,
            unmatched[:5],
        )
    for column, key_to_value in projectable.items():
        obs[column] = obs_keys.map(key_to_value)

    if set_index is not None:
        _apply_sdrf_set_index(obs, projectable, set_index, modality=modality)

    return list(projectable.keys())


def _resolve_projectable_columns(
    sdrf: pd.DataFrame, match_key: str, candidate_columns: list[str]
) -> tuple[dict[str, pd.Series], list[str]]:
    """Split candidate columns into {column: key->value map} for key-functional ones, and a skipped list."""
    grouped = sdrf.groupby(match_key, sort=False)
    key_representative = sdrf.drop_duplicates(match_key).set_index(match_key)

    projectable: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for column in candidate_columns:
        if column == match_key:
            continue
        if column not in sdrf.columns:
            raise ValueError(f"SDRF column not found: {column}.")
        if bool((grouped[column].nunique(dropna=False) <= 1).all()):
            projectable[column] = key_representative[column]
        else:
            skipped.append(column)
    return projectable, skipped


def _apply_sdrf_set_index(
    obs: pd.DataFrame, projectable: dict[str, pd.Series], set_index: str, *, modality: str
) -> None:
    if set_index in projectable:
        new_index = pd.Index(obs.index).map(projectable[set_index])
    elif set_index in obs.columns:
        new_index = pd.Index(obs[set_index])
    else:
        raise ValueError(
            f"set_index '{set_index}' is neither a projectable SDRF column nor an obs column (modality '{modality}')."
        )

    new_index = pd.Index(new_index)
    if new_index.isna().any():
        raise ValueError(f"set_index '{set_index}' has unmatched (NaN) values in {modality}.obs; cannot index.")
    if new_index.has_duplicates:
        raise ValueError(f"set_index '{set_index}' is not unique across {modality}.obs; cannot index.")
    obs.index = new_index.rename(None)


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


def _detect_metadata_input_kind(
    metadata: pd.DataFrame | str | PathLike[str],
) -> Literal["dataframe", "path", "url"]:
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


def _infer_source_metadata_format(
    source: str | Path,
) -> Literal["csv", "tsv", "parquet", "sdrf"] | None:
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
