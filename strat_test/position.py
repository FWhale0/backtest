from __future__ import annotations

from typing import Literal, Tuple, Any, Type
import pandas as pd

DfOrSeries = pd.DataFrame | pd.Series


class Position:
    """
    Class to calculate the position of a trading strategy.

    Parameters:
    - factor (DfOrSeries | float | int):
    Series or DataFrame representing the factor data.
    - ls (str): The method to select the long and short positions.
    - weight (Literal["equal", "value"] | pd.DataFrame):
    "equal" or "value". Default is "equal".

    Attributes:
    - factor (DfOrSeries | float | int):
    Series or DataFrame representing the factor data.
    - mode (Literal["abs", "rank", "quantile"]): The method to calculate the position.
    - ls (str): "long" or "short".
    - weight (Literal["equal", "value"] | pd.DataFrame): "equal" or "value".
    - posi (DfOrSeries): Series or DataFrame representing the position.
    """

    def __init__(
        self,
        factor: DfOrSeries,
        mode: Literal["abs", "rank", "quantile"],
        ls: str | float | int,
        weight: Literal["equal", "value"] | pd.DataFrame = "equal",
    ) -> None:
        # Check the input data
        self._validate_factor(factor)
        self._validate_mode(mode)
        self._validate_ls(ls, mode)
        self._validate_weight(weight)

        self.factor = factor
        self.mode = mode
        self.ls = ls
        self.weight = weight

        self.uw_posi_long, self.uw_posi_short = self._gen_posi()
        self.posi_long, self.posi_short = self._gen_weighted_posi()

    def _validate_factor(self, factor: DfOrSeries) -> None:
        """
        Check the factor data.
        """
        if isinstance(factor, (pd.Series, pd.DataFrame)):
            if isinstance(factor, pd.Series):
                self.factor = factor.to_frame()
        else:
            raise ValueError("The input factor must be a pandas Series or DataFrame.")

    def _validate_mode(self, mode: Literal["abs", "rank", "quantile"]) -> None:
        """
        Check the mode.
        """
        if mode not in ["abs", "rank", "quantile"]:
            raise ValueError("The mode must be 'abs', 'rank', or 'quantile'.")

    def _validate_ls(
        self,
        ls: str | float | int,
        mode: Literal["abs", "rank", "quantile"],
    ) -> None:
        """
        Check the long and short positions.
        """
        if not isinstance(ls, (str, float, int)):
            raise ValueError("The ls must be a string, float, or int.")
        if mode == "rank":
            if isinstance(ls, (float, int)) and ls <= 0:
                raise ValueError("The ls must be a positive number.")
        if mode == "quantile":
            if isinstance(ls, (float, int)) and (ls <= 0 or ls >= 1):
                raise ValueError("The ls must be a number between 0 and 1.")

    def _validate_weight(
        self, weight: Literal["equal", "value"] | pd.DataFrame
    ) -> None:
        """
        Check the weight.
        """
        if not isinstance(weight, (str, pd.DataFrame)):
            raise ValueError("The weight must be 'equal', 'value', or a DataFrame.")

    def _gen_posi(self) -> Tuple[DfOrSeries, DfOrSeries]:
        """
        Generate the position.
        """
        if self.mode == "abs":
            return self._gen_posi_abs()
        if self.mode == "rank":
            return self._gen_posi_rela("rank")
        if self.mode == "quantile":
            return self._gen_posi_rela("quantile")

        raise ValueError("The mode must be 'abs', 'rank', or 'quantile'.")

    def _get_ls_range(
        self, data_type: Type
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Process range from ls string.
        """
        ranges: dict[str, Any] = {"l": None, "s": None}

        assert isinstance(self.ls, str)
        if self.ls == "":
            raise ValueError("The ls string must not be empty.")

        ls_list = self.ls.split("_")  # get the rules
        if len(ls_list) > 2:
            raise ValueError(f"There must be less than one '_' in ls string: {self.ls}")

        for sub_rule in ls_list:
            if not sub_rule.startswith(("l:", "s:")):
                raise ValueError(f"Each rule must start with 'l:' or 's:': {sub_rule}")
            if len(sub_rule) <= 2:
                raise ValueError(f"Rule not specified: {sub_rule}")

            direction = sub_rule[0]  # get the direction of the rule
            if ranges[direction] is not None:
                raise ValueError(f"Duplicate rule: {self.ls}")

            rule_range = sub_rule[2:]
            nodes = rule_range.split("~")  # get the time nodes
            nodes = [data_type(node) for node in nodes]

            if len(nodes) > 2:
                raise ValueError(f"Multiple '-' in rule: {sub_rule}")

            # the start and end of the range
            if len(nodes) == 1:
                ranges[direction] = (nodes[0], nodes[0])
            else:
                ranges[direction] = (nodes[0], nodes[1])

        return ranges["l"], ranges["s"]

    def _gen_posi_abs(self) -> Tuple[DfOrSeries, DfOrSeries]:
        """
        Generate the position using the absolute value.
        """
        posi_long = self.factor.copy()
        posi_short = self.factor.copy()
        if isinstance(self.ls, (float, int)):
            # long the factor >= ls, short the factor <= ls
            # TODO: Using >= and <= may cause problems when the factor equals ls
            posi_long = self.factor >= self.ls
            posi_short = self.factor <= self.ls

        else:
            long_range, short_range = self._get_ls_range(float)
            # TODO: Using >= and <= may cause problems when the factor equals nodes
            long_min, long_max = long_range
            short_min, short_max = short_range
            posi_long = (long_min <= self.factor) & (self.factor <= long_max)
            posi_short = (short_min <= self.factor) & (self.factor <= short_max)

        return posi_long, posi_short

    def _gen_posi_rela(
        self, mode: Literal["rank", "quantile"]
    ) -> Tuple[DfOrSeries, DfOrSeries]:
        """
        Generate the position using the rank.
        """
        if mode == "rank":
            data_type, rank_pct = int, False
        elif mode == "quantile":
            data_type, rank_pct = float, True
        else:
            raise ValueError("The mode must be 'rank' or 'quantile'.")

        posi_long = pd.DataFrame(index=self.factor.index, columns=self.factor.columns)
        posi_short = pd.DataFrame(index=self.factor.index, columns=self.factor.columns)
        for date in self.factor.index:
            cross_as = self.factor.loc[date].rank(pct=rank_pct, ascending=True)
            cross_des = self.factor.loc[date].rank(pct=rank_pct, ascending=False)

            if isinstance(self.ls, data_type):
                # long the rank in the top ls, short the rank in the bottom ls
                # TODO: <= and >= may cause problems if some factors have the same rank
                posi_long.loc[date] = self.factor.loc[date][cross_des <= self.ls]
                posi_short.loc[date] = self.factor.loc[date][cross_as <= self.ls]
            elif isinstance(self.ls, str):
                long_range, short_range = self._get_ls_range(data_type)
                posi_long.loc[date] = self.factor.loc[date][
                    (cross_des >= long_range[0]) & (cross_des <= long_range[1])
                ]
                posi_short.loc[date] = self.factor.loc[date][
                    (cross_as >= short_range[0]) & (cross_as <= short_range[1])
                ]
            else:
                raise ValueError(f"ls must be {data_type} or str when mode is {mode}.")

        return posi_long, posi_short

    def _gen_weighted_posi(self) -> Tuple[DfOrSeries, DfOrSeries]:
        """
        Generate the weighted position.
        """
        posi_long = self.uw_posi_long
        posi_short = self.uw_posi_short

        if self.weight == "equal":
            posi_long = self.uw_posi_long / self.uw_posi_long.abs().sum()
            posi_short = self.uw_posi_short / self.uw_posi_short.abs().sum()
        elif self.weight == "value":
            posi_long = self.uw_posi_long / self.uw_posi_long.abs().sum()
            posi_short = self.uw_posi_short / self.uw_posi_short.abs().sum()

        if isinstance(self.weight, pd.DataFrame):
            posi_long = self.uw_posi_long * self.weight
            posi_short = self.uw_posi_short * self.weight

        return posi_long, posi_short
