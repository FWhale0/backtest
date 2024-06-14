from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sortedcontainers import SortedList
from typing import Literal


def calc_days(data: pd.Series) -> int:
    """
    Calculate the number of days in the given data.

    Args:
        data: A pandas Series representing the data.

    Returns:
        The number of days in the data.
    """
    return len(data)


def calc_accuracy(nworth: pd.Series) -> float:
    """
    Calculate the accuracy of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The accuracy of the net worth series.
    """
    return (nworth.pct_change() > 0).mean()


def calc_return(nworth: pd.Series) -> float:
    """
    Calculate the return of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The return of the net worth series.
    """
    return nworth.iloc[-1] / nworth.iloc[0] - 1


def calc_std(nworth: pd.Series) -> float:
    """
    Calculate the standard deviation of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The standard deviation of the net worth series.
    """
    return nworth.pct_change().std() * (calc_days(nworth) ** 0.5)


def calc_max_drawdown(nworth: pd.Series) -> float:
    """
    Calculate the maximum drawdown of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The maximum drawdown of the net worth series.
    """
    return (1 - nworth / nworth.cummax()).max()


def calc_calmar(nworth: pd.Series) -> float:
    """
    Calculate the Calmar ratio of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The Calmar ratio of the net worth series.
    """
    return calc_return(nworth) / calc_max_drawdown(nworth)


def calc_sharpe(nworth: pd.Series) -> float:
    """
    Calculate the Sharpe ratio of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The Sharpe ratio of the net worth series.
    """
    return calc_return(nworth) / calc_std(nworth)


def calc_turnover(posi: pd.DataFrame) -> float:
    """
    Calculate the turnover of the position DataFrame.

    Args:
        posi: A pandas DataFrame representing the position.

    Returns:
        The turnover of the position DataFrame.
    """
    return abs(posi.diff()).sum().sum() / 2 / calc_days(posi)


def calc_long(posi: pd.DataFrame) -> float:
    """
    Calculate the long position of the position DataFrame.

    Args:
        posi: A pandas DataFrame representing the position.

    Returns:
        The long position of the position DataFrame.
    """
    return posi[posi > 0].sum().sum() / calc_days(posi)


def calc_short(posi: pd.DataFrame) -> float:
    """
    Calculate the short position of the position DataFrame.

    Args:
        posi: A pandas DataFrame representing the position.

    Returns:
        The short position of the position DataFrame.
    """
    return abs(posi[posi < 0].sum().sum()) / calc_days(posi)


def calc_longshort(posi: pd.DataFrame) -> float:
    """
    Calculate the long and short position of the position DataFrame.

    Args:
        posi: A pandas DataFrame representing the position.

    Returns:
        The sum of long and short position of the position DataFrame.
    """
    return calc_long(posi) + calc_short(posi)


def yearly_return(nworth: pd.Series) -> float:
    """
    Calculate the yearly return of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The yearly return of the net worth series.
    """
    return (calc_return(nworth) + 1) ** (252 / calc_days(nworth)) - 1


def yearly_vol(nworth: pd.Series) -> float:
    """
    Calculate the yearly volatility of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The yearly volatility of the net worth series.
    """
    return nworth.pct_change().std() * (252**0.5)


def yearly_calmar(nworth: pd.Series) -> float:
    """
    Calculate the yearly Calmar ratio of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The yearly Calmar ratio of the net worth series.
    """
    return yearly_return(nworth) / calc_max_drawdown(nworth)


def yearly_sharpe(nworth: pd.Series) -> float:
    """
    Calculate the yearly Sharpe ratio of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The yearly Sharpe ratio of the net worth series.
    """
    return yearly_return(nworth) / yearly_vol(nworth)


def gen_drawdown(nworth: pd.Series) -> pd.Series:
    """
    Generate the drawdown series of the net worth series.

    Args:
        nworth: A pandas Series representing the net worth.

    Returns:
        The drawdown series of the net worth series.
    """
    max_asset = nworth.cummax()
    return nworth / max_asset - 1


def scale(
    x: pd.DataFrame | pd.Series,
    how: Literal["minmax", "standard", "1stvalue", "divstd"] = "standard",
) -> pd.DataFrame:
    """
    Scale the input data using the specified method.

    Args:
        x: A pandas DataFrame or Series representing the input data.
        how: The scaling method to use.
        Options are "minmax", "standard", "1stvalue", or "divstd".

    Returns:
        The scaled data as a pandas DataFrame.
    """
    assert how in [
        "minmax",
        "standard",
        "1stvalue",
        "divstd",
    ], "how must be 'minmax', 'standard', '1stvalue', or 'divstd'"

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if how == "minmax":
        scaler = MinMaxScaler()
        scaled_x = scaler.fit_transform(x)
        return pd.DataFrame(scaled_x, columns=x.columns, index=x.index)
    if how == "standard":
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(x)
        return pd.DataFrame(scaled_x, columns=x.columns, index=x.index)
    if how == "1stvalue":
        return x.apply(lambda x: x / abs(x.loc[x.first_valid_index()]))
    if how == "divstd":
        return x.apply(lambda x: x / x.std())


def yoy(
    x: pd.Series,
    freq: Literal["m", "d", "w"],
) -> pd.Series:
    """
    Calculate the year-over-year (YoY) change of the input series.

    Args:
        x: A pandas Series representing the input series.
        freq: The frequency of the input series.
        Options are "m" (monthly), "d" (daily), or "w" (weekly).

    Returns:
        The YoY change of the input series.
    """
    if freq == "m":
        x_m = x.resample("M").last()
        x_yoy = x_m.pct_change(12)
    if freq == "d":
        x_d = x.resample("D").last()
        x_yoy = x_d.pct_change(365)
    if freq == "w":
        x_w = x.resample("W").last()
        x_yoy = x_w.pct_change(52)

    x_yoy = x_yoy.reindex(x.index, method="ffill")

    return x_yoy


def sma(
    x: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the simple moving average (SMA) of the input series.

    Args:
        x: A pandas Series representing the input series.
        window: The window size for calculating the SMA.

    Returns:
        The SMA of the input series.
    """
    return x.rolling(window).mean()


def ema(
    x: pd.Series,
    alpha: float,
    adjust: bool = False,
) -> pd.Series:
    """
    Calculate the exponential moving average (EMA) of the input series.

    Args:
        x: A pandas Series representing the input series.
        alpha: The smoothing factor for calculating the EMA.
        adjust: Whether to adjust the EMA calculation.

    Returns:
        The EMA of the input series.
    """
    return x.ewm(alpha=alpha, adjust=adjust).mean()


def ts_rank(
    s: pd.Series,
    window: int,
    method: Literal["pandas", "numpy", "dynamic"] = "dynamic",
) -> pd.Series:
    """
    Calculate the rank of the input series within a rolling window.

    Args:
        s: A pandas Series representing the input series.
        window: The window size for calculating the rank.
        method: The method to use for calculating the rank.
        Options are "pandas", "numpy", or "dynamic".

    Returns:
        The rank of the input series within the rolling window.
    """
    s = s.dropna()
    if method == "pandas":
        rankings = s.rolling(window).apply(lambda x: x.rank().iloc[-1])
        return (rankings - 1) / (window - 1)
    if method == "numpy":
        find_rank = lambda x: np.searchsorted(np.sort(x), x[-1]) + 1
        rankings = s.rolling(window).apply(find_rank)
        return (rankings - 1) / (window - 1)
    if method == "dynamic":
        n = len(s)
        rankings = np.full(n, np.nan)
        sorted_list = SortedList()

        # Initialize the sorted list with the first window elements
        for i in range(window):
            sorted_list.add(s.iloc[i])

        # Sliding window
        for i in range(window, n):
            # Remove the element that is sliding out of the window
            sorted_list.remove(s.iloc[i - window])
            # Add the new element that is entering the window
            sorted_list.add(s.iloc[i])

            # Compute the rank of the new element
            rankings[i] = sorted_list.index(s.iloc[i])

        return pd.Series(rankings / (window - 1), index=s.index).dropna()
