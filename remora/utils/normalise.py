import pandas as pd
import numpy as np


class normaliser():

    def __init__(self, type):

        self.type = type.lower()

    def fit(self, x):
        if self.type == 'minmax':
            self.maximum = x.to_numpy().max()
            self.minimum = x.to_numpy().min()


    @property
    def _max(self):
        return self.maximum

    @property
    def _min(self):
        return self.minimum

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            #return (x - x.to_numpy().min())/(x.to_numpy().max() - x.to_numpy().min())
            return (x - self._min)/(self._max - self._min)

        else:
            return (x - min(x))/(max(x) - min(x))
