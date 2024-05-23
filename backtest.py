import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union as U


class Backtest:
    def __init__(
        self,
        factor: U[pd.DataFrame, pd.Series, None] = None,
        asset: U[pd.DataFrame, pd.Series, None] = None,
        long: U[int, float, None] = None,
        short: U[int, float, None] = None,
        lsway: Literal["rate", "num"] = None,
        long_fee: float = 0.001,
        short_fee: float = 0.001,
    ):
        self.factor = factor
        self.asset = asset
        self.long = long
        self.short = short
        self.lsway = lsway
        self.long_fee = long_fee
        self.short_fee = short_fee

    def IC(self) -> pd.DataFrame:
        return self.factor.corrwith(self.asset, axis=0, method="spearman")