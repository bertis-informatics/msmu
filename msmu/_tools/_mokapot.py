import mokapot
import pandas as pd


class Mokapot:
    def _sage2pin(self) -> pd.DataFrame: ...

    def _run_mokapot(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def _extract_q_values(self, df: pd.DataFrame) -> pd.DataFrame: ...
