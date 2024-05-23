import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union


class NetWorthMaker:
    def __init__(
        self,
        price: Union[pd.DataFrame, pd.Series],
        position: Union[pd.DataFrame, pd.Series],
    ):
        self.price = price
        self.position = position

        assert self._check_data_match(), "Data mismatch"
        self.networth = self._calc_networth()

    def _check_data_match(self) -> bool:
        dropped_posi = self.position.dropna(how="all", axis=1)
        dropped_posi = dropped_posi.dropna(how="all", axis=0)

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

    def _calc_networth(self) -> pd.Series:
        start = self.position.first_valid_index()
        end = self.position.last_valid_index()
        price = self.price.loc[start:end]
        position = self.position.loc[start:end]

        nworth = pd.Series(index=price.index, name="net_worth")
        nworth.iloc[0] = 1
        hold = 0

        for d in price.index[1:]:
            last_index = price.index.get_loc(d) - 1
            last_price = price.iloc[last_index]
            last_asset = nworth.iloc[last_index]

            nworth.loc[d] = last_asset + hold * (price.loc[d] - last_price)

            if position.loc[d] > 0 and hold <= 0:
                hold = nworth.loc[d] / price.loc[d]
            elif position.loc[d] < 0 and hold >= 0:
                hold = -nworth.loc[d] / price.loc[d]

        return nworth
