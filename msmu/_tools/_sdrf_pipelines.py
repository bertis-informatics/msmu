from __future__ import annotations

import logging
from importlib import metadata
from pathlib import Path

import pandas as pd


def validate_sdrf_dataframe(
    sdrf: pd.DataFrame,
    *,
    source: str | Path | None = None,
    template: str = "ms-proteomics",
    skip_ontology: bool = True,
) -> None:
    """
    Validate an SDRF DataFrame with the installed ``sdrf-pipelines`` package.

    Validation stays inside the tools boundary and operates on DataFrames
    instead of file paths.
    """
    if not isinstance(sdrf, pd.DataFrame):
        raise TypeError("sdrf must be a pandas DataFrame.")

    try:
        metadata.version("sdrf-pipelines")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "SDRF validation requires the optional 'sdrf-pipelines' package. Install it to use validate_sdrf=True."
        ) from exc

    from sdrf_pipelines.sdrf.schemas import SchemaRegistry, SchemaValidator
    from sdrf_pipelines.sdrf.sdrf import SDRFDataFrame

    validator = SchemaValidator(SchemaRegistry())
    errors = validator.validate(
        SDRFDataFrame(sdrf.copy()),
        template,
        use_ols_cache_only=False,
        skip_ontology=skip_ontology,
    )
    fatal_errors = [error for error in errors if getattr(error, "error_type", None) == logging.ERROR]
    if not fatal_errors:
        return

    seen: set[str] = set()
    messages: list[str] = []
    for error in fatal_errors:
        message = getattr(error, "message", None) or str(error)
        if message in seen:
            continue
        seen.add(message)
        messages.append(message)

    subject = str(source) if source is not None else "DataFrame"
    raise ValueError(f"SDRF validation failed for {subject}: {'; '.join(messages)}")
