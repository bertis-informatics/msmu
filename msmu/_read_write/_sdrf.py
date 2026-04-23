from __future__ import annotations

import contextlib
import csv
import io
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import mudata as md
import pandas as pd


_STRUCTURED_HEADER_RE = re.compile(r"^(characteristics|comment|factor value)\[(.+)\]$", re.IGNORECASE)
_TMT_LABEL_RE = re.compile(r"(?:TMT)?(1[0-9]{2}[NC]?)", re.IGNORECASE)
_PATH_SUFFIXES = {".raw", ".mzml", ".mzxml", ".wiff", ".d"}
# Deterministic scalar delimiter for obs values collapsed from multiple SDRF fraction rows.
_COLLAPSED_VALUE_DELIMITER = " | "
_REMOTE_SDRF_SCHEMES = {"http", "https"}
_SDRF_URL_TIMEOUT_SECONDS = 30
_MAX_SDRF_DOWNLOAD_BYTES = 100 * 1024 * 1024
_SDRF_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def read_sdrf(
    sdrf_file: str | Path,
    *,
    validate: bool = True,
    skip_ontology: bool = True,
) -> pd.DataFrame:
    """
    Read an SDRF file into obs-ready metadata.

    The returned DataFrame keeps one row per SDRF row and exposes core MAGE-TAB
    proteomics fields using stable snake_case column names. Duplicate SDRF
    headers are preserved by suffixing repeated output columns with ``_2``,
    ``_3``, and so on.
    """
    source = str(sdrf_file)
    with _materialized_sdrf_path(sdrf_file) as path:
        if validate:
            validate_sdrf_file(path, skip_ontology=skip_ontology)

        raw_df, raw_headers = _read_tabular_sdrf(path)
    metadata_df = _normalise_sdrf_dataframe(raw_df, raw_headers)
    metadata_df.attrs["sdrf_file"] = source
    metadata_df.attrs["sdrf_raw"] = _sdrf_source_table_payload(raw_df, raw_headers)

    return metadata_df


def validate_sdrf_file(sdrf_file: str | Path, *, skip_ontology: bool = True) -> None:
    """
    Validate an SDRF file with sdrf-pipelines when it is installed.

    ``sdrf-pipelines`` is intentionally optional. This loader uses the installed
    ``parse_sdrf`` console-script entry point in-process where possible. Some
    versions expose a plain console-script function instead of the Click command
    itself, so subprocess execution is retained as a fallback.
    """
    try:
        metadata.version("sdrf-pipelines")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "SDRF validation requires the optional 'sdrf-pipelines' package. " "Install it to use validate_sdrf=True."
        ) from exc

    args = ["validate-sdrf", "--sdrf_file", str(sdrf_file)]
    if skip_ontology:
        args.append("--skip-ontology")

    result = _run_parse_sdrf(args)
    if skip_ontology and result.exit_code != 0 and "No such option" in result.output:
        result = _run_parse_sdrf(["validate-sdrf", "--sdrf_file", str(sdrf_file)])

    if result.exit_code != 0:
        message = result.output.strip() or repr(result.exception)
        raise ValueError(f"SDRF validation failed for {sdrf_file}: {message}")


def attach_sdrf_metadata(
    mdata: md.MuData,
    sdrf_file: str | Path | None,
    *,
    validate: bool = True,
    skip_ontology: bool = True,
) -> md.MuData:
    if sdrf_file is None:
        return mdata

    sdrf_metadata = read_sdrf(sdrf_file, validate=validate, skip_ontology=skip_ontology)
    return merge_sdrf_metadata(mdata, sdrf_metadata, sdrf_file=sdrf_file)


@contextlib.contextmanager
def _materialized_sdrf_path(sdrf_file: str | Path) -> Iterator[Path]:
    source = str(sdrf_file)
    parsed = urllib.parse.urlparse(source)

    if "://" in source and parsed.scheme in _REMOTE_SDRF_SCHEMES:
        with tempfile.TemporaryDirectory(prefix="msmu-sdrf-") as temp_dir:
            path = Path(temp_dir) / "remote.sdrf.tsv"
            _download_sdrf_url(source, path)
            yield path
        return

    if parsed.scheme and "://" in source:
        supported = ", ".join(sorted(_REMOTE_SDRF_SCHEMES))
        raise ValueError(f"Unsupported SDRF URL scheme '{parsed.scheme}'. Supported schemes are: {supported}.")

    yield Path(sdrf_file)


def _download_sdrf_url(
    url: str,
    destination: Path,
    *,
    timeout: int | None = None,
    max_bytes: int | None = None,
) -> None:
    timeout = _SDRF_URL_TIMEOUT_SECONDS if timeout is None else timeout
    max_bytes = _MAX_SDRF_DOWNLOAD_BYTES if max_bytes is None else max_bytes

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is not None and status >= 400:
                raise ValueError(f"Failed to download SDRF URL {url}: HTTP status {status}.")

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    expected_size = int(content_length)
                except ValueError:
                    expected_size = None
                if expected_size is not None and expected_size > max_bytes:
                    raise ValueError(_sdrf_download_size_error(url, expected_size, max_bytes))

            total_bytes = 0
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(_SDRF_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ValueError(_sdrf_download_size_error(url, total_bytes, max_bytes))
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        raise OSError(f"Failed to download SDRF URL {url}: HTTP status {exc.code} {exc.reason}.") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise OSError(f"Failed to download SDRF URL {url}: {reason}.") from exc
    except OSError as exc:
        raise OSError(f"Failed to download SDRF URL {url}: {exc}.") from exc


def _sdrf_download_size_error(url: str, actual_bytes: int, max_bytes: int) -> str:
    return (
        f"SDRF URL {url} exceeds maximum download size of {max_bytes} bytes "
        f"({actual_bytes} bytes received or advertised)."
    )


def merge_sdrf_metadata(
    mdata: md.MuData,
    sdrf_metadata: pd.DataFrame,
    *,
    sdrf_file: str | Path | None = None,
) -> md.MuData:
    if not isinstance(mdata, md.MuData):
        raise TypeError("mdata must be a MuData object.")
    if sdrf_metadata.empty:
        raise ValueError("SDRF metadata is empty.")

    match_summary: dict[str, dict[str, object]] = {}
    metadata_columns = list(sdrf_metadata.columns)

    for mod in mdata.mod.keys():
        adata = mdata.mod[mod]
        matched_obs, matched_column, matched_count = _match_metadata_to_obs(adata.obs.index, sdrf_metadata)
        adata.obs = _merge_obs_metadata(adata.obs, matched_obs)
        adata.uns["sdrf"] = {
            "path": None if sdrf_file is None else str(sdrf_file),
            "matched_column": matched_column,
            "matched_obs": matched_count,
        }
        match_summary[str(mod)] = {
            "matched_column": matched_column,
            "matched_obs": matched_count,
            "n_obs": int(adata.n_obs),
        }

    global_metadata = pd.DataFrame(index=mdata.obs.index)
    for mod in mdata.mod.keys():
        mod_obs = mdata.mod[mod].obs.reindex(global_metadata.index)
        for col in metadata_columns:
            if col not in mod_obs.columns:
                continue
            if col not in global_metadata.columns:
                global_metadata[col] = mod_obs[col]
            else:
                global_metadata[col] = global_metadata[col].combine_first(mod_obs[col])

    mdata.obs = _merge_obs_metadata(mdata.obs, global_metadata)
    sdrf_uns = {
        "path": None if sdrf_file is None else str(sdrf_file),
        "columns": _sdrf_columns_uns_entry(sdrf_metadata.attrs.get("sdrf_columns", [])),
        "collapse_delimiter": _COLLAPSED_VALUE_DELIMITER,
        "matched_modalities": match_summary,
    }
    raw_sdrf = _sdrf_source_table_uns_entry(sdrf_metadata.attrs.get("sdrf_raw"))
    if raw_sdrf is not None:
        sdrf_uns["raw"] = raw_sdrf
        sdrf_uns["table"] = raw_sdrf["table"]
        sdrf_uns["original_headers"] = raw_sdrf["original_headers"]

    mdata.uns["sdrf"] = sdrf_uns

    return mdata


def _load_parse_sdrf_entrypoint():
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        candidates = entry_points.select(group="console_scripts", name="parse_sdrf")
    else:
        candidates = [ep for ep in entry_points.get("console_scripts", []) if ep.name == "parse_sdrf"]

    candidates = list(candidates)
    if not candidates:
        raise RuntimeError("Installed 'sdrf-pipelines' package does not expose a 'parse_sdrf' entry point.")

    return candidates[0].load()


@dataclass(frozen=True)
class _ValidationResult:
    exit_code: int
    output: str
    exception: BaseException | None = None


def _run_parse_sdrf(args: list[str]) -> _ValidationResult:
    try:
        parse_sdrf = _load_parse_sdrf_entrypoint()
    except Exception as exc:
        result = _run_parse_sdrf_subprocess(args)
        if result is not None:
            return result
        raise RuntimeError("Installed 'sdrf-pipelines' package could not load the 'parse_sdrf' entry point.") from exc

    result = _run_parse_sdrf_entrypoint(parse_sdrf, args)
    if result is not None:
        return result

    result = _run_parse_sdrf_subprocess(args)
    if result is not None:
        return result

    raise RuntimeError(
        "Installed 'sdrf-pipelines' package exposes 'parse_sdrf', but it could not be invoked as a Click command "
        "and no 'parse_sdrf' executable was found on PATH."
    )


def _run_parse_sdrf_entrypoint(parse_sdrf: Any, args: list[str]) -> _ValidationResult | None:
    try:
        import click
    except ImportError:
        return None

    command = _coerce_click_command(parse_sdrf, click)
    if command is not None:
        return _invoke_click_command(command, args)

    return _call_parse_sdrf_function(parse_sdrf, args, click)


def _invoke_click_command(command: Any, args: list[str]) -> _ValidationResult:
    from click.testing import CliRunner

    result = CliRunner().invoke(command, args)
    return _ValidationResult(result.exit_code, result.output, result.exception)


def _coerce_click_command(parse_sdrf: Any, click: Any) -> Any | None:
    if isinstance(parse_sdrf, click.Command):
        return parse_sdrf

    named_candidates = list(_named_entrypoint_click_command_candidates(parse_sdrf))
    other_candidates = list(_other_entrypoint_click_command_candidates(parse_sdrf))
    for candidate in [*named_candidates, *other_candidates]:
        if isinstance(candidate, click.Group):
            return candidate

    for candidate in named_candidates:
        if isinstance(candidate, click.Command):
            return candidate

    return None


def _named_entrypoint_click_command_candidates(parse_sdrf: Any) -> Iterable[Any]:
    module_globals = getattr(parse_sdrf, "__globals__", {})
    for name in ("cli", "main", "parse_sdrf"):
        candidate = module_globals.get(name)
        if candidate is not parse_sdrf:
            yield candidate


def _other_entrypoint_click_command_candidates(parse_sdrf: Any) -> Iterable[Any]:
    module_globals = getattr(parse_sdrf, "__globals__", {})
    for candidate in module_globals.values():
        if candidate is not parse_sdrf:
            yield candidate


def _call_parse_sdrf_function(parse_sdrf: Any, args: list[str], click: Any) -> _ValidationResult | None:
    if not callable(parse_sdrf):
        return None

    try:
        from click.testing import CliRunner
    except ImportError:
        return None

    stdout = io.StringIO()
    stderr = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["parse_sdrf", *args]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                returned = parse_sdrf()
            except SystemExit as exc:
                exit_code = _normalise_exit_code(exc.code)
                return _ValidationResult(
                    exit_code, stdout.getvalue() + stderr.getvalue(), None if exit_code == 0 else exc
                )
            except click.ClickException as exc:
                exc.show(file=stderr)
                return _ValidationResult(exc.exit_code, stdout.getvalue() + stderr.getvalue(), exc)
            except Exception:
                return None
    finally:
        sys.argv = old_argv

    if isinstance(returned, click.Command):
        result = CliRunner().invoke(returned, args)
        return _ValidationResult(
            result.exit_code, stdout.getvalue() + result.output + stderr.getvalue(), result.exception
        )

    if isinstance(returned, int):
        return _ValidationResult(returned, stdout.getvalue() + stderr.getvalue())

    return None


def _normalise_exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _run_parse_sdrf_subprocess(args: list[str]) -> _ValidationResult | None:
    executable = shutil.which("parse_sdrf")
    if executable is None:
        return None

    completed = subprocess.run(
        [executable, *args],
        capture_output=True,
        check=False,
        text=True,
    )
    return _ValidationResult(completed.returncode, completed.stdout + completed.stderr)


def _read_tabular_sdrf(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            raw_headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(raw_headers):
                raise ValueError(
                    f"{path} row {line_number} has {len(row)} fields, expected {len(raw_headers)} from the header."
                )
            rows.append(row)

    if not raw_headers:
        raise ValueError(f"{path} does not contain an SDRF header.")

    raw_df = pd.DataFrame(rows, columns=raw_headers)
    return raw_df, raw_headers


def _normalise_sdrf_dataframe(raw_df: pd.DataFrame, raw_headers: list[str]) -> pd.DataFrame:
    output_columns: dict[str, pd.Series] = {}
    output_counts: dict[str, int] = {}
    column_metadata: list[dict[str, object]] = []

    for position, raw_header in enumerate(raw_headers):
        parsed = _parse_sdrf_header(raw_header)
        if parsed is None:
            continue

        base_column, kind, key = parsed
        output_counts[base_column] = output_counts.get(base_column, 0) + 1
        occurrence = output_counts[base_column]
        output_column = base_column if occurrence == 1 else f"{base_column}_{occurrence}"

        values = raw_df.iloc[:, position].replace("", pd.NA)
        output_columns[output_column] = values
        column_metadata.append(
            {
                "column": output_column,
                "original_header": raw_header,
                "kind": kind,
                "key": key,
                "occurrence": occurrence,
            }
        )

    missing = {"source_name", "assay_name"} - set(output_columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"SDRF file is missing required column(s): {missing_text}.")

    metadata_df = pd.DataFrame(output_columns, index=raw_df.index)
    metadata_df.attrs["sdrf_columns"] = column_metadata

    return metadata_df


def _sdrf_source_table_payload(raw_df: pd.DataFrame, raw_headers: list[str]) -> dict[str, object]:
    return {
        "rows": raw_df.to_numpy(dtype=object).tolist(),
        "original_headers": list(raw_headers),
    }


def _sdrf_source_table_uns_entry(raw_payload: object) -> dict[str, object] | None:
    if not isinstance(raw_payload, dict):
        return None

    rows = raw_payload.get("rows")
    original_headers = raw_payload.get("original_headers")
    if not isinstance(rows, list) or not isinstance(original_headers, list):
        return None

    headers_are_unique = len(set(original_headers)) == len(original_headers)
    table_columns = (
        original_headers if headers_are_unique else [f"column_{position}" for position in range(len(original_headers))]
    )
    table = pd.DataFrame(rows, columns=table_columns)
    return {
        "table": table,
        "original_headers": list(original_headers),
        "columns_are_original_headers": headers_are_unique,
    }


def _sdrf_columns_uns_entry(columns: object) -> pd.DataFrame:
    if not isinstance(columns, list):
        return pd.DataFrame(columns=["column", "original_header", "kind", "key", "occurrence"])

    column_table = pd.DataFrame(columns, columns=["column", "original_header", "kind", "key", "occurrence"])
    for col in ["column", "original_header", "kind", "key"]:
        column_table[col] = column_table[col].fillna("").astype(str)
    return column_table


def _parse_sdrf_header(raw_header: str) -> tuple[str, str, str | None] | None:
    normalised = _normalise_header(raw_header)
    if normalised == "source name":
        return "source_name", "source", None
    if normalised == "sample name":
        return "sample_name", "sample", None
    if normalised == "assay name":
        return "assay_name", "assay", None
    if normalised == "technology type":
        return "technology_type", "technology", None

    match = _STRUCTURED_HEADER_RE.match(normalised)
    if match is None:
        return None

    kind = match.group(1)
    key = match.group(2).strip()
    prefix = {"characteristics": "characteristics", "comment": "comment", "factor value": "factor_value"}[kind]

    return f"{prefix}_{_slugify(key)}", prefix, key


def _normalise_header(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug or "value"


def _candidate_columns(sdrf_metadata: pd.DataFrame) -> list[str]:
    candidates = ["assay_name", "sample_name", "comment_label", "comment_data_file", "source_name"]
    candidates.extend(col for col in sdrf_metadata.columns if col.startswith("comment_label_"))
    candidates.extend(col for col in sdrf_metadata.columns if col.startswith("comment_data_file_"))
    return [col for col in candidates if col in sdrf_metadata.columns]


def _match_metadata_to_obs(
    obs_index: pd.Index,
    sdrf_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, str, int]:
    best_metadata: pd.DataFrame | None = None
    best_column: str | None = None
    best_count = -1

    for candidate_column in _candidate_columns(sdrf_metadata):
        candidate_map = _make_candidate_map(sdrf_metadata[candidate_column])
        matched_indices: list[tuple[object, ...] | None] = []
        for obs_name in obs_index:
            matched_indices.append(_match_obs_name(obs_name, candidate_map))

        matched_count = sum(index is not None for index in matched_indices)
        if matched_count > best_count:
            best_count = matched_count
            best_column = candidate_column
            best_metadata = _metadata_for_matches(sdrf_metadata, obs_index, matched_indices)

    if best_metadata is None or best_column is None or best_count == 0:
        candidates = ", ".join(_candidate_columns(sdrf_metadata))
        raise ValueError(f"SDRF metadata could not be matched to AnnData obs using candidate columns: {candidates}.")

    return best_metadata, best_column, best_count


def _make_candidate_map(values: pd.Series) -> dict[str, tuple[object, ...]]:
    aliases_to_rows: dict[str, list[object]] = {}
    for row_index, value in values.items():
        for alias in _value_aliases(value):
            key = _normalise_match_key(alias)
            if key and row_index not in aliases_to_rows.setdefault(key, []):
                aliases_to_rows[key].append(row_index)

    return {alias: tuple(rows) for alias, rows in aliases_to_rows.items()}


def _match_obs_name(obs_name: object, candidate_map: dict[str, tuple[object, ...]]) -> tuple[object, ...] | None:
    for alias in _value_aliases(obs_name):
        row_indices = candidate_map.get(_normalise_match_key(alias))
        if row_indices:
            return row_indices
    return None


def _metadata_for_matches(
    sdrf_metadata: pd.DataFrame,
    obs_index: pd.Index,
    matched_indices: Iterable[tuple[object, ...] | None],
) -> pd.DataFrame:
    rows = []
    for row_indices in matched_indices:
        if row_indices is None:
            rows.append(pd.Series({col: pd.NA for col in sdrf_metadata.columns}))
        else:
            rows.append(_collapse_sdrf_rows(sdrf_metadata.loc[list(row_indices), :]))

    matched = pd.DataFrame(rows, index=obs_index)
    matched.index = obs_index
    return matched


def _collapse_sdrf_rows(rows: pd.DataFrame) -> pd.Series:
    collapsed = {}
    for col in rows.columns:
        collapsed[col] = _collapse_sdrf_values(rows[col])
    return pd.Series(collapsed)


def _collapse_sdrf_values(values: pd.Series) -> object:
    distinct_values: list[object] = []
    seen: set[str] = set()

    for value in values:
        if pd.isna(value):
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            distinct_values.append(value)

    if not distinct_values:
        return pd.NA
    if len(distinct_values) == 1:
        return distinct_values[0]

    # Keep obs scalar-valued when fraction rows disagree while preserving SDRF row order.
    return _COLLAPSED_VALUE_DELIMITER.join(str(value) for value in distinct_values)


def _merge_obs_metadata(obs: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = obs.copy()
    aligned_metadata = metadata_df.reindex(merged.index)
    for col in aligned_metadata.columns:
        if col in merged.columns:
            merged[col] = merged[col].combine_first(aligned_metadata[col])
        else:
            merged[col] = aligned_metadata[col]
    return merged


def _value_aliases(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    raw = str(value).strip()
    if not raw:
        return set()

    aliases = {raw}
    path_text = raw.replace("\\", "/")
    path_name = Path(path_text).name
    aliases.add(path_name)
    aliases.add(_strip_known_suffix(path_name))

    tmt_match = _TMT_LABEL_RE.search(raw.replace("-", ""))
    if tmt_match:
        label = tmt_match.group(1).upper()
        aliases.add(label)
        aliases.add(f"TMT{label}")

    return {alias for alias in aliases if alias}


def _strip_known_suffix(value: str) -> str:
    lowered = value.lower()
    for suffix in _PATH_SUFFIXES:
        if lowered.endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def _normalise_match_key(value: object) -> str:
    return str(value).strip().lower()
