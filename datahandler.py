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
            data = web.get_data_yahoo(self.ticker, self.startdate, self.enddate)
            data['1change'] = data['Adj Close'].diff(1)
            #remove unused columns
            data = data[['1change']]
            #remove rows with nan
            data = data[10:]
            #preprocess data
            data = data.apply(preprocessing.scale)
            data = data.apply(self.scale_vals)
            data.to_csv(self.filename)
            self.data = data

    def scale_vals(self, vals):
        #scale to {-0.8,0.8}
        self.vals_max = np.max(vals)
        self.vals_min = np.min(vals)
        scale = 1.6 / (self.vals_max - self.vals_min)
        outs = []
        for val in vals:
            outs.append((scale * (val - self.vals_min)) - 0.8)

        mean = np.mean(outs)
        for i, val in enumerate(outs):
            outs[i] = val - mean

        return np.array(outs)