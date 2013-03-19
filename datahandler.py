import csv, os.path, ystockquote
import pandas as pan
import numpy as np
import datetime
import matplotlib.pyplot as plt


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
        print type(self.values[0])
        self.dates = (datetime.datetime.strptime(s[0], "%Y-%m-%d") for s in self.data)
        self.series = pan.Series(self.values, pan.DatetimeIndex(self.dates))

        change_days = 1
        changes = self.series.diff(change_days)

        plt.figure()
        changes.plot()
        plt.show()