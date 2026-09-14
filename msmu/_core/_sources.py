"""In-memory URL inputs shared by readers and provenance within one call."""

from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
from urllib.parse import urlsplit
from urllib.request import urlopen

_sources = ContextVar("msmu_url_sources", default=None)


def is_url(value):
    return isinstance(value, str) and urlsplit(value).scheme.lower() in {"http", "https", "ftp"}


@contextmanager
def source_scope():
    if _sources.get() is not None:
        yield
        return
    buffers = {}
    token = _sources.set(buffers)
    try:
        yield
    finally:
        _sources.reset(token)
        for buffer in buffers.values():
            buffer.close()


def get_download_buffer(url):
    buffers = _sources.get()
    return buffers.get(url) if buffers is not None else None


@contextmanager
def open_source(source):
    if not is_url(source):
        yield source
        return
    with source_scope():
        buffers = _sources.get()
        if source not in buffers:
            buffer = BytesIO()
            try:
                with urlopen(source, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        buffer.write(chunk)
            except BaseException:
                buffer.close()
                raise
            buffers[source] = buffer
        buffer = buffers[source]
        buffer.seek(0)
        yield buffer
