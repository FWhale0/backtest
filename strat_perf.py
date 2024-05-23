import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

from backtest.calcfuncs import (
    calc_days,
    calc_accuracy,
    calc_return,
    calc_std,
    calc_max_drawdown,
    calc_calmar,
    calc_sharpe,
    calc_turnover,
    calc_long,
    calc_short,
    calc_longshort,
    yearly_return,
    yearly_vol,
    yearly_calmar,
    yearly_sharpe,
    gen_drawdown,
)


class StratPerf:
    def __init__(
        self,
        net_worth: pd.Series,
        posi: Union[pd.DataFrame, pd.Series],
        price: Union[pd.DataFrame, pd.Series, None] = None,
        baseline: Union[pd.Series, None] = None,
    ):
        assert net_worth.index.equals(posi.index), "Index mismatch"
        if price:
            assert net_worth.index.equals(price.index), "Index mismatch"
        if baseline:
            assert net_worth.index.equals(baseline.index), "Index mismatch"
        assert net_worth.notna().all().all(), "Asset contains NaN"
        assert posi.notna().all().all(), "Position contains NaN"

        self.net_worth = net_worth
        self.posi = posi
        self.price = price
        self.baseline = baseline
        self.perf = self._performance()

    def _performance(self) -> pd.DataFrame:
        nworth_g_y = self.net_worth.groupby(self.net_worth.index.year)
        posi_g_y = self.posi.groupby(self.posi.index.year)

        cols = [
            "days",
            "acc.%",
            "ret.%",
            "std.%",
            "mdd.%",
            "calmar",
            "sharpe",
            "tr.%",
            "long.%",
            "short.%",
            "abs.%",
        ]
        perf = pd.DataFrame(columns=cols)
        for y in nworth_g_y.groups.keys():
            nworth_y = nworth_g_y.get_group(y)
            posi_y = posi_g_y.get_group(y)

            perf.loc[y] = [
                calc_days(nworth_y),
                calc_accuracy(nworth_y),
                calc_return(nworth_y),
                calc_std(nworth_y),
                calc_max_drawdown(nworth_y),
                calc_calmar(nworth_y),
                calc_sharpe(nworth_y),
                calc_turnover(posi_y),
                calc_long(posi_y),
                calc_short(posi_y),
                calc_longshort(posi_y),
            ]

        perf.loc["total"] = [
            calc_days(self.net_worth),
            calc_accuracy(self.net_worth),
            yearly_return(self.net_worth),
            yearly_vol(self.net_worth),
            calc_max_drawdown(self.net_worth),
            yearly_calmar(self.net_worth),
            yearly_sharpe(self.net_worth),
            calc_turnover(self.posi),
            calc_long(self.posi),
            calc_short(self.posi),
            calc_longshort(self.posi),
        ]

        return perf

    def _parse_toshow(self) -> pd.DataFrame:
        toshow = self.perf.copy()
        toshow["days"] = toshow["days"].astype(int)
        toshow["acc.%"] = (toshow["acc.%"] * 100).round(2)
        toshow["ret.%"] = (toshow["ret.%"] * 100).round(2)
        toshow["std.%"] = (toshow["std.%"] * 100).round(2)
        toshow["mdd.%"] = (toshow["mdd.%"] * 100).round(2)
        toshow["calmar"] = toshow["calmar"].round(2)
        toshow["sharpe"] = toshow["sharpe"].round(2)
        toshow["tr.%"] = (toshow["tr.%"] * 100).round(2)
        toshow["long.%"] = (toshow["long.%"] * 100).round(2)
        toshow["short.%"] = (toshow["short.%"] * 100).round(2)
        toshow["abs.%"] = (toshow["abs.%"] * 100).round(2)
        return toshow

    def get_all(self) -> pd.DataFrame:
        return self._parse_toshow()

    def get_total(self) -> pd.Series:
        return self._parse_toshow().loc["total"]

    def get_by_name(self, name: Union[str, list]) -> pd.Series:
        return self._parse_toshow().loc[name]

    def plot(self, figsize=(12, 4)):
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        self.net_worth.plot(ax=ax1, label="Asset")

        drawdown = gen_drawdown(self.net_worth)
        ax2.fill_between(
            drawdown.index, drawdown, 0, color="red", alpha=0.3, label="Drawdown"
        )
        ax2.set_ylim(-1, 0)
        if self.baseline:
            self.baseline.plot(ax=ax1, label="Baseline")

        # TODO: Find a better way to locate the legend
        ax1.legend()
        ax2.legend()

        plt.close(fig)  # TODO: Find a better way to prevent the plot from showing
        return fig
