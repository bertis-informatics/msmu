import logging
from enum import IntEnum


class LogLevel(IntEnum):
    NOTSET = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def setup_logger(level: LogLevel = LogLevel.INFO) -> logging.Logger:
    logger = logging.getLogger("msmu")
    logger.setLevel(level)

    if not any(getattr(h, "_msmu_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
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
                handler.setLevel(level)

    return logger
