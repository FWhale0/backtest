from __future__ import annotations

from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from strat_test.position import Position
from strat_test.net_worth import NetWorth
from strat_test.strat_perf import StratPerf

DfOrSeries = Union[pd.DataFrame, pd.Series]


class Portfolio:
    def __init__(
        self,
        price: DfOrSeries,
        pred: DfOrSeries,
        volume: DfOrSeries,
    ):
        self.price = price.resample("1h").last()
        self.factor = pred.shift(-1).drop(columns=["end_time"], errors="ignore")
        self.used_factor = self.factor.copy()
        self.volume = volume

    def volume_filter(self, threshold):
        volume = self.volume.resample("1h").last()
        liquidity_mask = volume.shift(1).rolling(24).sum() > threshold
        self.used_factor = self.factor[liquidity_mask]

    def position(self, mode: str, ls: str):
        factor, price = self.used_factor.align(self.price, axis=0, join="inner")
        factor[price.isna() | price == 0] = np.nan
        self.posi = Position(factor, mode, ls)

        return self.posi

    def net_worth(
        self,
        posi: Position,
        fee: float,
        ignore_posi_exceed: bool,
        method: str,
    ):
        nworth = NetWorth(self.price, posi, fee, ignore_posi_exceed, method)
        return nworth

    def backtest(
        self,
        mode: str,
        ls: str,
        fee: float = 0,
        ignore_posi_exceed: bool = False,
        method: str = "vector",
    ):
        self.posi = self.position(mode, ls)
        self.nworth = self.net_worth(self.posi, fee, ignore_posi_exceed, method)
        return self.nworth
