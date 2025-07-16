import numpy as np

from sklearn.preprocessing import StandardScaler


# Adapted from maplight
# https://github.com/maplightrx/MapLight-TDC/blob/c249378c63232354d17083c83fe94fe728960a27/maplight.py#L15
class LogStandardScaler:
    def __init__(self, log=False):
        self.log = log
        self.offset = 0.0
        self.scaler = StandardScaler()

    def fit(self, y_in):
        y = y_in

        if self.log:
            self.offset = np.min([np.min(y), 0.0])
            y = y - self.offset
            y = np.log10(y + 1.0)

        self.scaler.fit(y)

    def transform(self, y_in):
        y = y_in

        if self.log:
            y = y - self.offset
            y = np.log10(y + 1.0)

        y = self.scaler.transform(y)

        return y

    def fit_transform(self, y_in):
        y = self.fit(y_in)
        return self.transform(y)

    def inverse_transform(self, y_in):
        y = self.scaler.inverse_transform(y_in)

        if self.log:
            y = 10.0**y - 1.0
            y = y + self.offset

        return y
