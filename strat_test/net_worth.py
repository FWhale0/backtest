from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union
import warnings
from tqdm.auto import tqdm

from strat_test.strat_perf import StratPerf
from strat_test.position import Position


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
        position: pd.DataFrame | pd.Series | Position,
        fee: float = 0.0,
        ignore_posi_exceed: bool = False,
        method: Literal["vector", "progress"] = "vector",
    ):
        # Convert Series to DataFrame
        if isinstance(price, pd.Series):
            price = price.to_frame()
        if isinstance(position, pd.Series):
            position = position.to_frame()
        if isinstance(position, Position):
            position = position.get_posi()
        self.price = price
        self.position = position
        self.fee = fee

        # Check if the data match
        assert self._check_data_match(), "Data mismatch"
        assert self._check_posi_legal(
            ignore_posi_exceed
        ), "Absolute value of position should be less than or equal to 1"

        # Calculate net worth and performance
        self.networth = self._calc_networth(method)
        self.perf = None

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
        if self.position.abs().sum(axis=1).gt(1.00000001).any():
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

        # TODO: ignore when 1-column DataFrame
        if not dropped_posi.columns.isin(self.price.columns).all():
            if len(dropped_posi.columns) == 1:
                self.position.columns = self.price.columns
                dropped_posi.columns = self.price.columns
            else:
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
        p_return = price.pct_change().to_numpy(dtype=float)
        posi_shifted = position.shift().fillna(0).to_numpy(dtype=float)
        posi_change = np.diff(posi_shifted, axis=0)
        posi_change = np.vstack([np.zeros(posi_change.shape[1]), posi_change])

        # Calculate transaction cost
        tran_cost = np.abs(posi_change) * self.fee

        # Calculate net worth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            net_return = p_return * posi_shifted - tran_cost
        nworth = (np.nansum(net_return, axis=1) + 1).cumprod()

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

        last_hold = nworth.iloc[0] * 0
        for d in tqdm(price.index[1:], leave=False):
            last_index = price.index.get_loc(d) - 1
            last_asset = nworth.iloc[last_index]
            last_price = price.iloc[last_index]

            hold = ((last_asset * posi.loc[d]) / last_price).fillna(0)
            tran_cost = np.abs(hold - last_hold) * last_price * self.fee
            nworth.loc[d] = last_asset + (hold * (price.loc[d] - last_price)).sum() - tran_cost.sum()
            last_hold = hold

        return nworth

    def _calc_networth(self, method: Literal["vector", "progress"]) -> pd.Series:
        """
        Calculate the net worth of the strategy.

        Returns:
            pd.Series: The net worth of the strategy.

        """
        if method == "progress":
            return self._calc_networth_progress()
        elif method == "vector":
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
        if self.perf is None:
            self.perf = StratPerf(self.networth)
        return self.perf.plot(figsize=figsize)

    def total_perf(self) -> float:
        if self.perf is None:
            self.perf = StratPerf(self.networth)
        return self.perf.get_total()

    def annual_perf(self) -> float:
        if self.perf is None:
            self.perf = StratPerf(self.networth)
        return self.perf.get_annual()

    def single_networth(self, col: str) -> pd.Series:
        if len(self.position.columns) == 1:
            raise ValueError("Only one column in the position data")
        return NetWorth(self.price[col], self.position[col], self.fee)