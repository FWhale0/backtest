import pandas as pd


def calc_days(data: pd.Series) -> int:
    return len(data)


def calc_accuracy(nworth: pd.Series) -> float:
    return (nworth.pct_change() > 0).mean()


def calc_return(nworth: pd.Series) -> float:
    return nworth.iloc[-1] / nworth.iloc[0] - 1


def calc_std(nworth: pd.Series) -> float:
    return nworth.pct_change().std() * (calc_days(nworth) ** 0.5)


def calc_max_drawdown(nworth: pd.Series) -> float:
    return (1 - nworth / nworth.cummax()).max()


def calc_calmar(nworth: pd.Series) -> float:
    return calc_return(nworth) / calc_max_drawdown(nworth)


def calc_sharpe(nworth: pd.Series) -> float:
    return calc_return(nworth) / calc_std(nworth)


def calc_turnover(posi: pd.DataFrame) -> float:
    return abs(posi.diff()).sum().sum() / 2 / calc_days(posi)


def calc_long(posi: pd.DataFrame) -> float:
    return posi[posi > 0].sum().sum() / calc_days(posi)


def calc_short(posi: pd.DataFrame) -> float:
    return abs(posi[posi < 0].sum().sum()) / calc_days(posi)


def calc_longshort(posi: pd.DataFrame) -> float:
    return calc_long(posi) + calc_short(posi)


def yearly_return(nworth: pd.Series) -> float:
    return calc_return(nworth) * 252 / calc_days(nworth)


def yearly_vol(nworth: pd.Series) -> float:
    return nworth.pct_change().std() * (252**0.5)


def yearly_calmar(nworth: pd.Series) -> float:
    return yearly_return(nworth) / calc_max_drawdown(nworth)


def yearly_sharpe(nworth: pd.Series) -> float:
    return yearly_return(nworth) / yearly_vol(nworth)


def gen_drawdown(nworth: pd.Series) -> pd.Series:
    max_asset = nworth.cummax()
    return nworth / max_asset - 1

