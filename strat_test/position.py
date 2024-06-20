from __future__ import annotations

from typing import Literal, Tuple, Any, Type, Union
import pandas as pd

DfOrSeries = Union[pd.DataFrame, pd.Series]


class Position:
    """
    A class representing a position in a trading strategy.

    Parameters:
        factor_data (DfOrSeries): The factor data used for generating positions.
        mode (Literal["abs", "rank", "quantile"]): The mode for generating positions.
        ls (str | float | int): The long/short threshold for generating positions.
        weight (Literal["equal", "value"] | pd.DataFrame, optional): The weighting scheme for positions. Defaults to "equal".

    Attributes:
        factor (DfOrSeries): The factor data used for generating positions.
        mode (Literal["abs", "rank", "quantile"]): The mode for generating positions.
        ls (str | float | int): The long/short threshold for generating positions.
        weight (Literal["equal", "value"] | pd.DataFrame): The weighting scheme for positions.
        uw_posi_long (DfOrSeries): The unweighted long positions.
        uw_posi_short (DfOrSeries): The unweighted short positions.
        posi (DfOrSeries): The weighted positions.

    Raises:
        ValueError: If the input factor is not a pandas Series or DataFrame.
        ValueError: If the mode is not one of "abs", "rank", or "quantile".
        ValueError: If the ls is not a string, float, or int.
        ValueError: If the ls is a negative number when mode is "rank".
        ValueError: If the ls is not a number between 0 and 1 when mode is "quantile".
        ValueError: If the weight is not one of "equal", "value", or a DataFrame.

    """

    def __init__(
        self,
        factor_data: DfOrSeries,
        mode: Literal["abs", "rank", "quantile"],
        ls: str | float | int,
        weight: Literal["equal", "value"] | pd.DataFrame = "equal",
    ) -> None:
        # Check the input data
        self._validate_factor(factor_data)
        self._validate_mode(mode)
        self._validate_ls(ls, mode)
        self._validate_weight(weight)

        self.factor = factor_data
        self.mode = mode
        self.ls = ls
        self.weight = weight

        self.uw_posi_long, self.uw_posi_short = self._gen_posi()
        self.posi = self._gen_weighted_posi()

    def _validate_factor(self, factor: DfOrSeries) -> None:
        """
        Validate the input factor data.

        Args:
            factor (DfOrSeries): The input factor data.

        Raises:
            ValueError: If the input factor is not a pandas Series or DataFrame.
        """
        if isinstance(factor, (pd.Series, pd.DataFrame)):
            if isinstance(factor, pd.Series):
                self.factor = factor.to_frame()
        else:
            raise ValueError("The input factor must be a pandas Series or DataFrame.")

    def _validate_mode(self, mode: Literal["abs", "rank", "quantile"]) -> None:
        """
        Validate the input mode.

        Args:
            mode (Literal["abs", "rank", "quantile"]): The input mode.

        Raises:
            ValueError: If the mode is not one of "abs", "rank", or "quantile".
        """
        if mode not in ["abs", "rank", "quantile"]:
            raise ValueError("The mode must be 'abs', 'rank', or 'quantile'.")

    def _validate_ls(
        self,
        ls: str | float | int,
        mode: Literal["abs", "rank", "quantile"],
    ) -> None:
        """
        Validate the input ls.

        Args:
            ls (str | float | int): The input ls.
            mode (Literal["abs", "rank", "quantile"]): The mode for generating positions.

        Raises:
            ValueError: If the ls is not a string, float, or int.
            ValueError: If the ls is a negative number when mode is "rank".
            ValueError: If the ls is not a number between 0 and 1 when mode is "quantile".
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
        Validate the input weight.

        Args:
            weight (Literal["equal", "value"] | pd.DataFrame): The input weight.

        Raises:
            ValueError: If the weight is not one of "equal", "value", or a DataFrame.
        """
        if not isinstance(weight, (str, pd.DataFrame)):
            raise ValueError("The weight must be 'equal', 'value', or a DataFrame.")

    def _gen_posi(self) -> Tuple[DfOrSeries, DfOrSeries]:
        """
        Generate unweighted long and short positions based on the mode and ls.

        Returns:
            Tuple[DfOrSeries, DfOrSeries]: The unweighted long and short positions.

        Raises:
            ValueError: If the mode is not one of "abs", "rank", or "quantile".
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
        Get the long and short range from the ls string.

        Args:
            data_type (Type): The data type for parsing the ls range.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: The long and short range.

        Raises:
            ValueError: If the ls string is empty.
            ValueError: If there is more than one '_' in the ls string.
            ValueError: If a rule does not start with 'l:' or 's:'.
            ValueError: If a rule is not specified.
            ValueError: If a rule has multiple '-'.
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
        Generate unweighted long and short positions based on the absolute mode.

        Returns:
            Tuple[DfOrSeries, DfOrSeries]: The unweighted long and short positions.
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
        Generate unweighted long and short positions based on the relative mode.

        Args:
            mode (Literal["rank", "quantile"]): The relative mode.

        Returns:
            Tuple[DfOrSeries, DfOrSeries]: The unweighted long and short positions.

        Raises:
            ValueError: If the mode is not one of "rank" or "quantile".
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
                posi_long.loc[date] = cross_des <= self.ls
                posi_short.loc[date] = cross_as <= self.ls
            elif isinstance(self.ls, str):
                long_range, short_range = self._get_ls_range(data_type)
                l_min, l_max = long_range
                s_min, s_max = short_range
                posi_long.loc[date] = (cross_des >= l_min) & (cross_des <= l_max)
                posi_short.loc[date] = (cross_as >= s_min) & (cross_as <= s_max)
            else:
                raise ValueError(f"ls must be {data_type} or str when mode is {mode}.")

        return posi_long, posi_short

    def _gen_weighted_posi(self) -> DfOrSeries:
        """
        Generate weighted positions based on the weighting scheme.

        Returns:
            DfOrSeries: The weighted positions.
        """
        if self.weight == "equal":
            posi = self.uw_posi_long - self.uw_posi_short
            posi = posi.div(posi.abs().sum(axis=1), axis=0)
        elif self.weight == "value":
            posi_long = self.uw_posi_long / self.uw_posi_long.abs().sum()
            posi_short = self.uw_posi_short / self.uw_posi_short.abs().sum()

        if isinstance(self.weight, pd.DataFrame):
            posi_long = self.uw_posi_long * self.weight
            posi_short = self.uw_posi_short * self.weight

        return posi
