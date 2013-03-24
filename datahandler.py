import os.path
import pandas as pan
import pandas.io.data as web
import numpy as np
import datetime


class DataHandler():
    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    data = []

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
            data['1lag'] = data['1change'].shift(1)
            data['2lag'] = data['1change'].shift(2)
            data['3lag'] = data['1change'].shift(3)
            data['4lag'] = data['1change'].shift(4)
            #remove rows with nan
            data = data[10:]
            #preprocess data
            data = data.apply(self.scale_vals)
            print data.head(10)
            data.to_csv(self.filename)
            self.data = data

    @staticmethod
    def scale_vals(vals):
        #log transform to reduce dynamic range and outliers
        outs = []
        for val in vals:
            if val >= 0:
                outs.append(np.log(np.abs(val) + 1))
            else:
                outs.append(-np.log(np.abs(val) + 1))

        #scale to {-0.8,0.8}
        vals_max = np.max(outs)
        vals_min = np.min(outs)
        scale = 1.6 / (vals_max - vals_min)
        for i, val in enumerate(outs):
            outs[i] = (scale * (val - vals_min)) - 0.8

        #mean to 0
        mean = np.mean(outs)
        for i, val in enumerate(outs):
            outs[i] = val - mean

        return np.array(outs)