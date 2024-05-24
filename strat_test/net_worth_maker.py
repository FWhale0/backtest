import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union

from strat_test.strat_perf import StratPerf


class NetWorthMaker:
    def __init__(
        self,
        price: Union[pd.DataFrame, pd.Series],
        position: Union[pd.DataFrame, pd.Series],
        force: bool = False,
    ):
        self.price = price
        self.position = position

        assert self._check_data_match(), "Data mismatch"
        assert self._check_posi_legal(
            force
        ), "Absolute value of position should be less than or equal to 1"
        self.networth = self._calc_networth()
        self.perf = StratPerf(self.networth, position, price)

    def _check_posi_legal(self, force: bool) -> bool:
        if force:
            return True

        # The absolute value of position should be less than or equal to 1
        if self.position.abs().sum(axis=1).round(8).gt(1).any():
            return False
        return True

    def _check_data_match(self) -> bool:
        dropped_posi = self.position.dropna(how="all", axis=1)
        dropped_posi = dropped_posi.dropna(how="all", axis=0)
        dropped_posi = dropped_posi.replace(0, np.nan)

        if not dropped_posi.columns.isin(self.price.columns).all():
            return False

        if not dropped_posi.index.isin(self.price.index).all():
            return False

        for col in dropped_posi.columns:
            posi_col = dropped_posi[col]
            posi_col = posi_col.dropna()
            if self.price.loc[posi_col.index, col].isna().any():
                return False

        return True

    def _calc_networth_vector(self) -> pd.Series:
        start = self.position.first_valid_index()
        end = self.position.last_valid_index()
        price = self.price.loc[start:end]
        position = self.position.loc[start:end]
        index = price.index

        # Use return to calculate net worth
        p_return = price.pct_change().to_numpy()
        position = position.shift().to_numpy()

        nworth = (np.nansum(p_return * position, axis=1) + 1).cumprod()

        return pd.Series(data=nworth, index=index, name="net_worth")

    def _calc_networth_progress(self) -> pd.Series:
        start = self.position.first_valid_index()
        end = self.position.last_valid_index()
        price = self.price.loc[start:end]
        posi = self.position.loc[start:end].shift(1)
        posi = posi.replace(0, np.nan)

        nworth = pd.Series(index=price.index, name="net_worth")
        nworth.iloc[0] = 1

        for d in price.index[1:]:
            last_index = price.index.get_loc(d) - 1
            last_asset = nworth.iloc[last_index]
            last_price = price.iloc[last_index]
            hold = (last_asset * posi.loc[d]) / last_price
            nworth.loc[d] = last_asset + (hold * (price.loc[d] - last_price)).sum()

        return nworth

    def _calc_networth(self) -> pd.Series:
        return self._calc_networth_progress()
        # return self._calc_networth_vector()

    def get_networth(self) -> pd.Series:
        return self.networth

    def plot(self, figsize=(12, 4)):
        return self.perf.plot(figsize=figsize)
