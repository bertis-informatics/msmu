import io
import logging
import sys
import subprocess

from msmu.logging_utils import (
    PACKAGE_LOGGER_NAME,
    LogLevel,
    ensure_null_handler,
    get_logger,
    setup_logger,
    prune_closed_package_stream_handlers,
)


def test_get_logger_returns_package_logger_by_default() -> None:
    logger = get_logger()
    assert logger.name == PACKAGE_LOGGER_NAME


def test_ensure_null_handler_adds_single_null_handler() -> None:
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = []
        ensure_null_handler()
        ensure_null_handler()
        null_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.NullHandler)]
        assert len(null_handlers) == 1
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_setup_logger_reuses_single_msmu_handler() -> None:
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = []
        configured = setup_logger(LogLevel.DEBUG)
        configured = setup_logger(LogLevel.INFO)

        msmu_handlers = [handler for handler in configured.handlers if getattr(handler, "_msmu_handler", False)]
        assert len(msmu_handlers) == 1
        assert configured.level == logging.INFO
        assert configured.propagate is False
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_setup_logger_replaces_closed_msmu_handler() -> None:
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        stale_handler._msmu_handler = True  # type: ignore[attr-defined]
        logger.handlers = [stale_handler]
        stream.close()

        configured = setup_logger(LogLevel.INFO)

        msmu_handlers = [handler for handler in configured.handlers if getattr(handler, "_msmu_handler", False)]
        assert len(msmu_handlers) == 1
        assert msmu_handlers[0] is not stale_handler
        configured.info("logger still works")
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_setup_logger_handler_recovers_when_stream_closes(monkeypatch) -> None:
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = []
        initial_stream = io.StringIO()
        monkeypatch.setattr(sys, "stderr", initial_stream)
        configured = setup_logger(LogLevel.INFO)
        handler = next(handler for handler in configured.handlers if getattr(handler, "_msmu_handler", False))
        initial_stream.close()

        replacement_stream = io.StringIO()
        monkeypatch.setattr(sys, "stderr", replacement_stream)
        prune_closed_package_stream_handlers()
        configured.info("logger recovered")

        assert "logger recovered" in replacement_stream.getvalue()
        assert handler.stream is replacement_stream
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_default_console_logging_without_public_setup_logger():
    result = subprocess.run(
        [sys.executable, "-c", "import msmu as mm; assert not hasattr(mm, 'setup_logger'); "
         "mm.logger.info('visible'); mm.logger.debug('hidden'); mm.logger.warning('notice')"],
        capture_output=True, text=True, check=True,
    )
    assert result.stderr.count("INFO - visible") == 1
    assert result.stderr.count("WARNING - notice") == 1
    assert "hidden" not in result.stderr
