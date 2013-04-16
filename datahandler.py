import os.path
import pandas as pan
import pandas.io.data as web
import numpy as np
import datetime
from pandas.tools.merge import merge
from pybrain.datasets import SequentialDataSet


class DataHandler():
    #Financial Data
    sp = ''
    filename = ''
    tickers = ''
    startdate = ''
    enddate = ''
    dataframe = pan.DataFrame()
    data = ''

    #fetch financial data from file or yahoo API
    def load_indices(self, tickers, startdate, lags):
        self.tickers = tickers
        self.filename = "DATA.csv"
        self.startdate = startdate
        self.enddate = datetime.date.today().strftime("%Y%m%d")
        if os.path.isfile(self.filename):
            data = pan.DataFrame.from_csv(self.filename)
            self.dataframe = data
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
                for i in range(1, lags + 1):
                    label = ticker + "%dlag" % i
                    data[label] = data[index].shift(i)
                #remove rows used for change calculation
                data = data[lags + 1:]
                print data.head(10)
                if ticker == "%5EGSPC":
                    self.sp = data
                else:
                    self.sp = merge(self.sp, data, left_index=True, right_index=True)
            self.dataframe = self.sp
            self.dataframe.to_csv(self.filename)

    def create_data(self, inputs, targets):
        data = SequentialDataSet(inputs, targets)
        for i in xrange(0, len(self.dataframe) - 1):
            data.newSequence()
            ins = self.dataframe.ix[i].values
            target = self.dataframe.ix[i + 1].values[0]
            data.appendLinked(ins, target)
        self.data = data

    def get_datasets(self, proportion):
        return self.data.splitWithProportion(proportion)


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