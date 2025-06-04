import mudata as md
import anndata as ad


class AnnData(ad.AnnData):
    pass


class AnnDataProxy:
    """Wrapper to preserve autocomplete for dynamic modalities like mdata.protein.obs"""

    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    def __getattr__(self, name):
        return getattr(self._adata, name)

    def __dir__(self):
        return dir(self._adata)

    def __repr__(self):
        return repr(self._adata)

    def __class__(self):
        return ad.AnnData


class MuData(md.MuData):
    _reserved_names = set(dir(md.MuData))

    def __getattr__(self, name: str):
        if name in self.mod and name not in self._reserved_names:
            return AnnDataProxy(self.mod[name])
        return super().__getattribute__(name)

    def __dir__(self):
        dynamic_attrs = [k for k in self.mod if k not in self._reserved_names]
        return sorted(set(super().__dir__()) | set(dynamic_attrs))

    def copy(self) -> "MuData":
        base_copy = super().copy()
        return MuData(
            data={key: self[key] for key in self.mod_names},
            uns=base_copy.uns,
            obs=base_copy.obs,
            var=base_copy.var,
        )
