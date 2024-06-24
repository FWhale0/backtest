from __future__ import annotations


class FacData:
    def __init__(self, factor):
        self.factor_data = factor.gen_data()

    def get_fac_data(self):
        return self.factor_data
