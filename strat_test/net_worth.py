from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union

from strat_test.strat_perf import StratPerf


class NetWorth:
    """
    Class to calculate net worth and performance of a trading strategy.

    Args:
        price (pd.DataFrame | pd.Series): The price data used for the strategy.
        position (pd.DataFrame | pd.Series): The position data used for the strategy.
        ignore_posi_exceed (bool, optional): Whether to ignore positions that exceed the absolute value of 1. Defaults to False.

    Attributes:
        price (pd.DataFrame | pd.Series): The price data used for the strategy.
        position (pd.DataFrame | pd.Series): The position data used for the strategy.
        networth (pd.Series): The net worth of the strategy.
        perf (StratPerf): The performance of the strategy.

    Raises:
        AssertionError: If there is a data mismatch or if the absolute value of position exceeds 1.

    """

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        position: pd.DataFrame | pd.Series,
        ignore_posi_exceed: bool = False,
    ):
        # Convert Series to DataFrame
        if isinstance(price, pd.Series):
            price = price.to_frame()
        if isinstance(position, pd.Series):
            position = position.to_frame()
        self.price = price
        self.position = position

        # Check if the data match
        assert self._check_data_match(), "Data mismatch"
        assert self._check_posi_legal(
            ignore_posi_exceed
        ), "Absolute value of position should be less than or equal to 1"

        # Calculate net worth and performance
        self.networth = self._calc_networth()
        self.perf = StratPerf(self.networth, position, price)

    def _check_posi_legal(self, ignore_posi_exceed: bool) -> bool:
        """
        Check if the positions are legal.

        Args:
            ignore_posi_exceed (bool): Whether to ignore positions
            that exceed the absolute value of 1.

        Returns:
            bool: True if positions are legal, False otherwise.

        """
        if ignore_posi_exceed:
            return True

        # The absolute value of position should be less than or equal to 1
        if self.position.abs().sum(axis=1).round(8).gt(1).any():
            return False
        return True

    def _check_data_match(self) -> bool:
        """
        Check if the price and position data match.

        Returns:
            bool: True if data matches, False otherwise.

        """
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
        """
        Calculate net worth using vectorized calculations.

        Returns:
            pd.Series: The net worth of the strategy.

        """
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
        """
        Calculate net worth using progressive calculations.

        Returns:
            pd.Series: The net worth of the strategy.

        """
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
        """
        Calculate the net worth of the strategy.

        Returns:
            pd.Series: The net worth of the strategy.

        """
        # return self._calc_networth_progress()
        return self._calc_networth_vector()

    def get_networth(self) -> pd.Series:
        """
        Get the net worth of the strategy.

        Returns:
            pd.Series: The net worth of the strategy.

        """
        return self.networth

    def plot(self, figsize=(12, 4)):
        """
        Plot the performance of the strategy.

        Args:
            figsize (tuple, optional): The size of the figure. Defaults to (12, 4).

        Returns:
            matplotlib.axes.Axes: The plot of the performance.

        """
        return self.perf.plot(figsize=figsize)

    def get_total(self) -> float:
        return self.perf.get_total()

    def get_annual(self) -> float:
        return self.perf.get_annual()