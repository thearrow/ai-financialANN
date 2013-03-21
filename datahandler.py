import os.path
import pandas as pan
import pandas.io.data as web
import numpy as np
import datetime
from sklearn import preprocessing


class DataHandler():
    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    data = []
    vals_min = 0
    vals_max = 0
    change_series_cache = {}
    val_series_cache = []

    #fetch financial data from file or yahoo API
    def load_index(self, ticker, startdate):
        self.ticker = ticker
        self.filename = "%s.csv" % ticker
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y%m%d")
        if os.path.isfile(self.filename):
            data = pan.DataFrame.from_csv(self.filename)
            self.data = data
        else:
            data = web.get_data_yahoo(self.ticker, self.startdate, self.enddate)['Adj Close']
            data.to_csv(self.filename)
            self.data = data

    def value_series(self):
        if len(self.val_series_cache) != 0:
            return self.val_series_cache
        else:
            #mean->0 ; stdev->
            norm_vals = preprocessing.scale(self.data.values.flatten())
            #max->1 ; min->-1
            norm_vals = self.scale_vals(norm_vals)
            self.val_series_cache = pan.Series(norm_vals, self.data.index)
            return self.val_series_cache

    def change_series(self, days):
        if days in self.change_series_cache.has_key:
            return self.change_series_cache.get(days)
        else:
            changes = self.data.diff(days)
            #mean->0 ; stdev->1
            change_vals = preprocessing.scale(changes.values[np.isfinite(changes.values)])
            #max->1 ; min->-1
            change_vals = self.scale_vals(change_vals)
            #pad change values with 0
            for i in range(0, days):
                change_vals = np.insert(change_vals, 0, 0.0)
            series = pan.Series(change_vals, changes.index)
            self.change_series_cache[days] = series
            return series

    def scale_vals(self, vals):
        #scale to {-1,1}
        self.vals_max = np.max(vals)
        self.vals_min = np.min(vals)
        scale = 2 / (self.vals_max - self.vals_min)
        outs = []
        for val in vals:
            outs.append(scale * val)
        return np.array(outs)