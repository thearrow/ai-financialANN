import csv, os.path, ystockquote
import pandas as pan
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
    dates = []
    values = []
    series = []
    vals_min = 0
    vals_max = 0

    #fetch financial data from file or yahoo API
    def load_index(self, ticker, startdate):
        self.ticker = ticker
        self.filename = "%s.csv" % ticker
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y%m%d")
        if os.path.isfile(self.filename):
            data = list(csv.reader(open(self.filename, 'rb'), delimiter=','))
            data.reverse()
            self.data = data
        else:
            data = ystockquote.get_historical_prices(self.ticker, self.startdate, self.enddate)
            data.pop(0)  # remove header line
            with open(self.filename, 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            data.reverse()
            self.data = data
        self.values = np.array(list(float(s[-1]) for s in self.data))
        self.dates = (datetime.datetime.strptime(s[0], "%Y-%m-%d") for s in self.data)
        self.series = pan.Series(self.values, pan.DatetimeIndex(self.dates))

    def value_series(self):
        #mean->0 ; stdev->1
        norm_vals = preprocessing.scale(self.series.values)
        #max->1 ; min->-1
        norm_vals = self.scale_vals(norm_vals)
        return pan.Series(norm_vals, self.series.index)

    def change_series(self, days):
        changes = self.series.diff(days)
        #mean->0 ; stdev->1
        change_vals = preprocessing.scale(changes.values[np.isfinite(changes.values)])
        #max->1 ; min->-1
        change_vals = self.scale_vals(change_vals)
        return pan.Series(change_vals, changes.index[days:])

    def scale_vals(self, vals):
        #scale to {-1,1}
        self.vals_max = np.max(vals)
        self.vals_min = np.min(vals)
        scale = 2 / (self.vals_max - self.vals_min)
        outs = []
        for val in vals:
            outs.append(scale * val)
        return np.array(outs)