from dataclasses import dataclass

import mudata as md
import numpy as np


@dataclass
class MuDataFlags:
    mod_names: list[str]


@dataclass
class AnnDataFlags:
    modality: str | None
    label: str | None
    aquisition: str | None
    has_purity: bool
    has_decoy: bool
    has_pep: bool
    has_var: bool
    has_quant: bool

    @property
    def acquisition(self) -> str | None:
        """Compatibility alias for the existing `aquisition` field."""
        return self.aquisition


class MuDataStatus:
    """Convenience view over modality-level MuData metadata flags."""

    def __init__(self, mdata: md.MuData):
        self._mdata: md.MuData = mdata
        self.set_mudata_flags()

        self.psm: AnnDataFlags | None = None
        self.peptide: AnnDataFlags | None = None
        self.protein: AnnDataFlags | None = None

        for mod_name in self.mod_names:
            self.set_anndata_flags(mod_name)

    def set_mudata_flags(self):
        self.mod_names = list(self._mdata.mod_names)

    def set_anndata_flags(self, mod_name: str):
        setattr(
            self,
            mod_name,
            AnnDataFlags(
                modality=mod_name,
                label=(
                    self._mdata.mod[mod_name].uns["label"]
                    if "label" in self._mdata.mod[mod_name].uns_keys()
                    else None
                ),
                aquisition=(
                    self._mdata.mod[mod_name].uns["acquisition"]
                    if "acquisition" in self._mdata.mod[mod_name].uns_keys()
                    else None
                ),
                has_purity="purity" in self._mdata.mod[mod_name].var.columns,
                has_decoy="decoy" in self._mdata.mod[mod_name].uns_keys(),
                has_pep="PEP" in self._mdata.mod[mod_name].var.columns,
                has_var=len(self._mdata.mod[mod_name].var) > 0,
                has_quant=~np.isnan(self._mdata.mod[mod_name].X).all(),
            ),
        )
