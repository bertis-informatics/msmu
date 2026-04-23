import logging
import sys
from enum import IntEnum
from typing import TextIO


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


class _MsmuStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        stream = self.stream
        if _is_closed_stream(stream):
            stream = _current_stderr_stream()
            if stream is None:
                return
            self.stream = stream

        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return

        try:
            stream.write(msg + self.terminator)
            self.flush()
        except (OSError, ValueError) as exc:
            if not _is_closed_stream_error(exc):
                self.handleError(record)
                return

            stream = _current_stderr_stream()
            if stream is None:
                return
            self.stream = stream
            try:
                stream.write(msg + self.terminator)
                self.flush()
            except (OSError, ValueError):
                return
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def _is_closed_stream(stream: object) -> bool:
    return bool(getattr(stream, "closed", False))


def _is_closed_stream_error(exc: OSError | ValueError) -> bool:
    return "closed" in str(exc).lower()


def _current_stderr_stream() -> TextIO | None:
    for stream in (sys.stderr, getattr(sys, "__stderr__", None)):
        if stream is not None and not _is_closed_stream(stream):
            return stream
    return None


def _has_closed_stream(handler: logging.Handler) -> bool:
    return _is_closed_stream(getattr(handler, "stream", None))


def prune_closed_stream_handlers(
    logger: logging.Logger | None = None,
    *,
    only_msmu_handlers: bool = True,
) -> logging.Logger:
    logger = get_logger() if logger is None else logger
    for handler in list(logger.handlers):
        if only_msmu_handlers and not getattr(handler, "_msmu_handler", False):
            continue
        if _has_closed_stream(handler):
            logger.removeHandler(handler)
            handler.close()
    return logger


def prune_closed_msmu_handlers(logger: logging.Logger | None = None) -> logging.Logger:
    return prune_closed_stream_handlers(logger, only_msmu_handlers=True)


def prune_closed_package_stream_handlers(package_name: str = PACKAGE_LOGGER_NAME) -> None:
    package_logger = logging.getLogger(package_name)
    prune_closed_stream_handlers(package_logger, only_msmu_handlers=False)

    logger_prefix = f"{package_name}."
    for logger_name, logger_obj in package_logger.manager.loggerDict.items():
        if not logger_name.startswith(logger_prefix) or not isinstance(logger_obj, logging.Logger):
            continue
        prune_closed_stream_handlers(logger_obj, only_msmu_handlers=False)


def setup_logger(
    level: LogLevel = LogLevel.INFO,
    *,
    propagate: bool = False,
) -> logging.Logger:
    logger = get_logger()
    prune_closed_msmu_handlers(logger)
    logger.setLevel(int(level))
    logger.propagate = propagate

    if not any(getattr(h, "_msmu_handler", False) for h in logger.handlers):
        handler = _MsmuStreamHandler()
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
