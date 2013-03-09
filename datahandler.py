import csv, numpy, os.path, ystockquote
from matplotlib import dates
from sklearn.preprocessing import MinMaxScaler
from datetime import date


class DataHandler():
    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    price_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    change_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    change2_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = []
    dates = []
    values = []

    #fetch financial data from file or yahoo API
    def load_index(self, ticker, startdate):
        self.ticker = ticker
        self.filename = "%s.csv" % ticker
        self.startdate = startdate
        self.enddate = date.today().strftime("%Y%m%d")
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
        self.values = list((float(s[-1]) for s in self.data))
        self.dates = list((dates.datestr2num(s[0]) for s in self.data))