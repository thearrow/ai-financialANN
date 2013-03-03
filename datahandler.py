import csv, numpy, os.path, ystockquote
from matplotlib import dates
from sklearn.preprocessing import MinMaxScaler


class DataHandler():

    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    scaler = MinMaxScaler(feature_range=(0,1),copy=False)
    data = []
    dates = []
    values = []

    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.filename = "%s.csv" % ticker
        self.startdate = start
        self.enddate = end
        self.load_index()

    #fetch financial data from file or yahoo API
    def load_index(self):
        if os.path.isfile(self.filename):
            data = list(csv.reader(open(self.filename,'rb'),delimiter=','))
            data.reverse()
            self.data = data
        else:
            data = ystockquote.get_historical_prices(self.ticker,self.startdate,self.enddate)
            data.pop(0)
            with open(self.filename, 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            data.reverse()
            self.data = data
        self.values = list((float(s[-1]) for s in self.data))
        self.dates = list((dates.datestr2num(s[0]) for s in self.data))
        self.normalize()

    #data normalization functions {0.1:0.9}
    def normalize(self):
        self.values = numpy.reshape(numpy.array(self.values),(len(self.values),1))
        self.scaler.fit_transform(self.values)
        self.values = list(numpy.reshape(self.values,(1,len(self.values))).flatten())

    def un_normalize(self):
        self.values = numpy.reshape(numpy.array(self.values),(len(self.values),1))
        self.scaler.inverse_transform(self.values)
        self.values = list(numpy.reshape(self.values,(1,len(self.values))).flatten())

    def get_dates(self):
        return self.dates

    def get_values(self):
        return self.values