class TmtLabelInfo:
    monoisotopic_mass = float()
    label = list()


class Tmt2(TmtLabelInfo):
    monoisotopic_mass: float = 225.155833
    label: list = ["126", "127"]


class Tmt6(TmtLabelInfo):
    monoisotopic_mass: float = 229.162932
    label: list = ["126", "127", "128", "129", "130", "131"]


class Tmt10(TmtLabelInfo):
    monoisotopic_mass: float = 229.162932
    label: list = [
        "126",
        "127N",
        "127C",
        "128N",
        "128C",
        "129N",
        "129C",
        "130N",
        "130C",
        "131",
    ]


class Tmt11(TmtLabelInfo):
    monoisotopic_mass: float = 229.162932
    label: list = [
        "126",
        "127N",
        "127C",
        "128N",
        "128C",
        "129N",
        "129C",
        "130N",
        "130C",
        "131N",
        "131C",
    ]


class Tmt16(TmtLabelInfo):
    monoisotopic_mass: float = 304.207146
    label: list = [
        "126",
        "127N",
        "127C",
        "128N",
        "128C",
        "129N",
        "129C",
        "130N",
        "130C",
        "131N",
        "131C",
        "132N",
        "132C",
        "133N",
        "133C",
        "134N",
    ]


class Tmt18(TmtLabelInfo):
    monoisotopic_mass: float = 304.207146
    label: list = [
        "126",
        "127N",
        "127C",
        "128N",
        "128C",
        "129N",
        "129C",
        "130N",
        "130C",
        "131N",
        "131C",
        "132N",
        "132C",
        "133N",
        "133C",
        "134N",
        "134C",
        "135N",
    ]


def to_sdrf_channel_label(reporter: str) -> str:
    """Reporter-ion label (e.g. ``"126"``, ``"127N"``) -> SDRF ``comment[label]`` channel (``"TMT126"``).

    SDRF-Proteomics spells every isobaric channel with the ``TMT`` prefix, plex-independently
    (TMTpro 16/18-plex reporters are ``TMT130N`` etc.), so readers emit obs names in this form to
    line up with SDRF ``comment[label]`` directly.
    """
    return f"TMT{reporter}"
