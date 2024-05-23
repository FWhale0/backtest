import pandas as pd

def calc_days(data: pd.Series) -> int:
    return len(data)

def calc_accuracy(asset: pd.Series) -> float:
    return (asset.pct_change() > 0).mean().values[0]

def calc_return(asset: pd.Series) -> float:
    return (asset.iloc[-1] / asset.iloc[0] - 1).values[0]

def calc_std(asset: pd.Series) -> float:
    return (asset.pct_change().std() * (calc_days(asset) ** 0.5)).values[0]

def calc_max_drawdown(asset: pd.Series) -> float:
    return (1 - asset / asset.cummax()).max().values[0]

def calc_calmar(asset: pd.Series) -> float:
    return calc_return(asset) / calc_max_drawdown(asset)

def calc_sharpe(asset: pd.Series) -> float:
    return calc_return(asset) / calc_std(asset)

def calc_turnover(posi: pd.DataFrame) -> float:
    return abs(posi.diff()).sum().sum() / 2 / calc_days(posi)

def calc_long(posi: pd.DataFrame) -> float:
    return posi[posi > 0].sum().sum() / calc_days(posi)

def calc_short(posi: pd.DataFrame) -> float:
    return posi[posi < 0].sum().sum() / calc_days(posi)

def calc_longshort(posi: pd.DataFrame) -> float:
    return calc_long(posi) + calc_short(posi)

def yearly_return(asset: pd.Series) -> float:
    return calc_return(asset) * 252 / calc_days(asset)

def yearly_vol(asset: pd.Series) -> float:
    return calc_std(asset) * (252 ** 0.5)

def yearly_calmar(asset: pd.Series) -> float:
    return yearly_return(asset) / calc_max_drawdown(asset)

def yearly_sharpe(asset: pd.Series) -> float:
    return yearly_return(asset) / yearly_vol(asset)