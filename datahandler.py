import os.path
import pandas as pan
import pandas.io.data as web
import numpy as np
import datetime
from pandas.tools.merge import merge


class DataHandler():
    #Financial Data
    sp = ''
    filename = ''
    tickers = ''
    startdate = ''
    enddate = ''
    data = pan.DataFrame()

    #fetch financial data from file or yahoo API
    def load_indices(self, tickers, startdate, threshold):
        self.tickers = tickers
        self.filename = "DATA.csv"
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y%m%d")
        if os.path.isfile(self.filename):
            data = pan.DataFrame.from_csv(self.filename)
            self.data = data
        else:
            for ticker in tickers:
                data = web.get_data_yahoo(ticker, self.startdate, self.enddate)
                index = ticker + '1change'
                data[index] = data['Adj Close'].pct_change(1)
                #remove unused columns and nan row
                data = data[[index]]
                data = data[1:]
                #filter out middle threshold noise
                #data = data[np.logical_or(data[index] >= threshold, data[index] <= -threshold)]
                #preprocess data
                data = data.apply(preprocess)
                #lag data
                lag1 = ticker + '1lag'
                lag2 = ticker + '2lag'
                lag3 = ticker + '3lag'
                lag4 = ticker + '4lag'
                data[lag1] = data[index].shift(1)
                data[lag2] = data[index].shift(2)
                data[lag3] = data[index].shift(3)
                data[lag4] = data[index].shift(4)
                #remove rows used for change calculation
                data = data[9:]
                print data.head(10)
                if ticker == "%5EGSPC":
                    self.sp = data
                else:
                    self.sp = merge(self.sp, data, left_index=True, right_index=True)
            self.data = self.sp
            self.data.to_csv(self.filename)


def preprocess(vals):
    #log transform to reduce dynamic range and outliers
    outs = []
    for val in vals:
        if val >= 0:
            outs.append(np.log(np.abs(val * 100) + 1))
        else:
            outs.append(-np.log(np.abs(val * 100) + 1))

    #scale to {-0.9,0.9}
    vals_max = np.max(outs)
    vals_min = np.min(outs)
    scale = 1.8 / (vals_max - vals_min)
    for i, val in enumerate(outs):
        outs[i] = (scale * (val - vals_min)) - 0.9

    #mean to 0
    mean = np.mean(outs)
    for i, val in enumerate(outs):
        outs[i] = val - mean

    return np.array(outs)