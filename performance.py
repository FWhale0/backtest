import pandas as pd
from typing import Union

from calcfuncs import (
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
)


class Performance:
    def __init__(self, asset: pd.Series, posi: pd.DataFrame):
        assert asset.index.equals(posi.index), "Index mismatch"
        assert asset.notna().all().all(), "Asset contains NaN"
        assert posi.notna().all().all(), "Position contains NaN"

        self.asset = asset
        self.posi = posi
        self.perf = self._performance()

    def _performance(self) -> pd.DataFrame:
        asset_g_y = self.asset.groupby(self.asset.index.year)
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
        for y in asset_g_y.groups.keys():
            asset_y = asset_g_y.get_group(y)
            posi_y = posi_g_y.get_group(y)

            perf.loc[y] = [
                calc_days(asset_y),
                calc_accuracy(asset_y),
                calc_return(asset_y),
                calc_std(asset_y),
                calc_max_drawdown(asset_y),
                calc_calmar(asset_y),
                calc_sharpe(asset_y),
                calc_turnover(posi_y),
                calc_long(posi_y),
                calc_short(posi_y),
                calc_longshort(posi_y),
            ]

        perf.loc["total"] = [
            calc_days(self.asset),
            calc_accuracy(self.asset),
            yearly_return(self.asset),
            yearly_vol(self.asset),
            calc_max_drawdown(self.asset),
            yearly_calmar(self.asset),
            yearly_sharpe(self.asset),
            calc_turnover(self.posi),
            calc_long(self.posi),
            calc_short(self.posi),
            calc_longshort(self.posi),
        ]

        return perf
    
    def get_all(self) -> pd.DataFrame:
        return self.perf
    
    def get_total(self) -> pd.Series:
        return self.perf.loc["total"]
    
    def get_by_name(self, name: Union[str, list]) -> pd.Series:
        if isinstance(name, list):
            return self.perf[name]
        else:
            return self.perf[name]