import logging
from enum import IntEnum


PACKAGE_LOGGER_NAME = "msmu"


class LogLevel(IntEnum):
    NOTSET = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    return logging.getLogger(name)


def ensure_null_handler() -> logging.Logger:
    logger = get_logger()
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def setup_logger(
    level: LogLevel = LogLevel.INFO,
    *,
    propagate: bool = False,
) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(int(level))
    logger.propagate = propagate

    if not any(getattr(h, "_msmu_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(int(level))
        handler._msmu_handler = True  # type: ignore[attr-defined]
        formatter = logging.Formatter(
            fmt="%(levelname)s - %(message)s",
            # datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if getattr(handler, "_msmu_handler", False):
                handler.setLevel(int(level))

    return logger
