import pandas as pd
import numpy as np


class minmaxNormaliser():

    #TODO: Add the normalisation methods that are generally used when analysing DNA/RNA reads

    def __init__(self, type):

        self.type = type.lower()

    def fit(self, x):
        if self.type == 'minmax':
            if isinstance(x, pd.DataFrame):
                self.maximum = x.to_numpy().max()
                self.minimum = x.to_numpy().min()

            elif isinstance(x, np.ndarray):
                self.maximum = x.max()
                self.minimum = x.min()

            elif isinstance(x, list):
                globMaxim = 0
                globMinim = 0
                for array in x:
                    maxim = max(array)
                    minim = min(array)
                    if maxim > globMaxim:
                        globMaxim = maxim
                    if minim < globMinim:
                        globMinim = minim

                self.maximum = globMaxim
                self.minimum = globMinim


    @property
    def _max(self):
        return self.maximum

    @property
    def _min(self):
        return self.minimum

    def transform(self, x):
        if isinstance(x, (pd.DataFrame, np.ndarray)):
            #return (x - x.to_numpy().min())/(x.to_numpy().max() - x.to_numpy().min())
            return (x - self._min)/(self._max - self._min)


        elif isinstance(x, list):
            for array in x:
                array = (array - self._min)/(self._max - self._min)
            return x
