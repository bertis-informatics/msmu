import logging

from msmu.logging_utils import PACKAGE_LOGGER_NAME, LogLevel, ensure_null_handler, get_logger, setup_logger


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
