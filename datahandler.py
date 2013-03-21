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
            data['1d-change'] = data['Adj Close'].pct_change(1)
            data['5d-change'] = data['Adj Close'].pct_change(5)
            data['10-mov-avg'] = pan.ewma(data['Adj Close'], com=4.5)
            data['20-mov-avg'] = pan.ewma(data['Adj Close'], com=9.5)
            #remove rows used to calculate initial moving averages
            data = data[20:]
            #remove unused columns
            data = data[['Adj Close', '1d-change', '5d-change', '10-mov-avg', '20-mov-avg']]
            #preprocess data
            data = data.apply(preprocessing.scale)
            data = data.apply(self.scale_vals)
            data.to_csv(self.filename)
            self.data = data

    def scale_vals(self, vals):
        #scale to {-1,1}
        self.vals_max = np.max(vals)
        self.vals_min = np.min(vals)
        scale = 2 / (self.vals_max - self.vals_min)
        outs = []
        for val in vals:
            outs.append(scale * val)
        return np.array(outs)