from ._sage_reader import Reader


class DiannReader(Reader):
    def __init__(self) -> None:
        super().__init__()
        self._search_engine = "Diann"
