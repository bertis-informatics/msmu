"""Content hashes; never hash provenance or densify sparse matrices."""

from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import struct

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pa = None

ALGORITHM = "sha256"


def compute_hash(value) -> str:
    """Hash supported data content. Unsupported objects raise TypeError.

    Unordered categories hash by values, allowing h5mu's string-to-category conversion.
    Numeric dtypes, axis order/names and ordered categorical metadata are significant.
    Files are streamed; sparse storage is canonical CSR and explicit zeroes are retained.
    """
    digest = sha256()

    def token(data):
        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)

    def visit_string_array(array):
        """Emit the existing string-array token stream using Arrow's native kernels."""
        token("array")
        visit((len(array),))
        token("values")
        chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
        for chunk in chunks:
            # Limit temporary buffers to 65,536 strings; boundaries never enter the hash.
            for start in range(0, len(chunk), 65536):
                binary = pc.cast(chunk.slice(start, 65536), pa.large_binary())
                lengths = pc.cast(pc.fill_null(pc.binary_length(binary), 0), pa.int64())
                length_bytes = pa.Array.from_buffers(
                    pa.binary(8), len(lengths), [None, lengths.buffers()[1]], offset=lengths.offset
                )
                encoded = pc.binary_join_element_wise(
                    pa.scalar(struct.pack("<Q", 3) + b"str", pa.large_binary()),
                    pc.cast(length_bytes, pa.large_binary()), binary,
                    pa.scalar(b"", pa.large_binary()),
                )
                encoded = pc.fill_null(encoded, struct.pack("<Q", 4) + b"null")
                offsets = np.frombuffer(encoded.buffers()[1], dtype="<i8", count=len(encoded) + 1,
                                        offset=encoded.offset * 8)
                digest.update(memoryview(encoded.buffers()[2])[offsets[0]:offsets[-1]])

    def visit(obj):
        if obj is None or obj is pd.NA or obj is pd.NaT:
            token("null")
        elif isinstance(obj, (bool, np.bool_)):
            token("bool")
            token(int(obj))
        elif isinstance(obj, (int, np.integer)):
            token("int")
            token(int(obj))
        elif isinstance(obj, (float, np.floating)):
            token("float")
            token("nan" if np.isnan(obj) else float(obj).hex())
        elif isinstance(obj, str):
            token("str")
            token(obj)
        elif isinstance(obj, bytes):
            token("bytes")
            token(obj)
        elif isinstance(obj, (datetime, date, np.datetime64, np.timedelta64)):
            token(type(obj).__name__)
            token(str(obj))
        elif isinstance(obj, BytesIO):
            token("file")
            with obj.getbuffer() as content:
                token(len(content))
                digest.update(content)
        elif isinstance(obj, Path):
            token("file")
            token(obj.stat().st_size)
            with obj.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        elif sparse.issparse(obj):
            token("sparse-csr")
            matrix = obj.tocsr(copy=True)
            matrix.sum_duplicates()
            matrix.sort_indices()
            visit(matrix.shape)
            visit(matrix.indptr.astype("<i8"))
            visit(matrix.indices.astype("<i8"))
            visit(matrix.data)
        elif isinstance(obj, pd.DataFrame):
            token("dataframe")
            visit(obj.index)
            visit(obj.columns)
            for index in range(obj.shape[1]):
                visit(obj.iloc[:, index].array)
        elif isinstance(obj, pd.Series):
            token("series")
            visit(obj.name)
            visit(obj.index)
            visit(obj.array)
        elif isinstance(obj, pd.MultiIndex):
            token("multiindex")
            visit(obj.names)
            visit(obj.tolist())
        elif isinstance(obj, pd.Index):
            token("index")
            visit(obj.name)
            visit(obj.array)
        elif isinstance(obj, pd.Categorical):
            if obj.ordered:
                token("ordered-category")
                visit(obj.categories)
            if pa is not None and isinstance(obj.categories.dtype, pd.StringDtype):
                visit_string_array(pc.cast(pa.array(obj), pa.large_string()))
            else:
                visit(np.asarray(obj))
        elif isinstance(obj, pd.api.extensions.ExtensionArray):
            # Nullable string/object/category storage is normalized by semantic values.
            if pd.api.types.is_numeric_dtype(obj.dtype):
                token(str(obj.dtype))
            if pa is not None and isinstance(obj.dtype, pd.StringDtype):
                visit_string_array(obj.__arrow_array__())
            else:
                visit(np.asarray(obj))
        elif isinstance(obj, np.ndarray):
            token("array")
            visit(obj.shape)
            if obj.dtype.kind in "biufcMm":
                token(obj.dtype.str.replace(">", "<"))
                array = np.asarray(obj, dtype=obj.dtype.newbyteorder("<"), order="C")
                # Normalize NaN payloads without changing the user's array.
                if obj.dtype.kind in "fc" and np.isnan(array).any():
                    array = array.copy()
                    if obj.dtype.kind == "c":
                        array.real[np.isnan(array.real)] = np.nan
                        array.imag[np.isnan(array.imag)] = np.nan
                    else:
                        array[np.isnan(array)] = np.nan
                data = memoryview(array).cast("B") if array.size else b""
                token(len(data))
                digest.update(data)
            elif obj.dtype.kind in "OUS":
                token("values")
                for item in obj.flat:
                    if item is None or item is pd.NA or item is pd.NaT or isinstance(item, float) and np.isnan(item):
                        visit(None)
                    else:
                        visit(item.item() if isinstance(item, np.generic) else item)
            else:
                raise TypeError(f"Unsupported array dtype: {obj.dtype}")
        elif isinstance(obj, (md.MuData, ad.AnnData)):
            token(type(obj).__name__)
            for attr in ("obs", "var", "obsm", "varm", "obsp", "varp"):
                token(attr)
                visit(getattr(obj, attr))
            token("uns")
            visit({k: v for k, v in obj.uns.items() if k != "_log"})
            if isinstance(obj, md.MuData):
                token("mod")
                visit(obj.mod)
                token("obsmap")
                visit(obj.obsmap)
                token("varmap")
                visit(obj.varmap)
            else:
                token("X")
                visit(obj.X)
                token("layers")
                visit({k: v for k, v in obj.layers.items() if k is not None})
                token("raw")
                if obj.raw is None:
                    visit(None)
                else:
                    visit(obj.raw.X)
                    visit(obj.raw.var)
                    visit(obj.raw.varm)
        elif isinstance(obj, Mapping):
            token("mapping")
            token(len(obj))
            if not all(isinstance(key, str) for key in obj):
                raise TypeError("Fingerprint mapping keys must be strings")
            for key in sorted(obj):
                token(key)
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            token("sequence")
            token(len(obj))
            for item in obj:
                visit(item)
        else:
            raise TypeError(f"Unsupported hash input type: {type(obj).__module__}.{type(obj).__qualname__}")

    visit(value)
    return digest.hexdigest()
