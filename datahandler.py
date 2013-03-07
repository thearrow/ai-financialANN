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
    changes = []
    changes2 = []

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
        self.changes = self.createChanges(1)
        self.changes2 = self.createChanges(2)
        self.changes = self.normalize(self.changes, self.change_scaler)
        self.changes2 = self.normalize(self.changes2, self.change2_scaler)
        self.values = self.normalize(self.values, self.price_scaler)

    def createChanges(self, days):
        changes = []
        for i in range(0, days):
            changes.append(0)
        for i in range(days, len(self.values)):
            changes.append(self.values[i] - self.values[i - days])
        return changes

    #data normalization (0.2,0.8)
    def normalize(self, input, scaler):
        data = numpy.reshape(numpy.array(input), (len(input), 1))
        scaler.fit_transform(data)
        data = list(numpy.reshape(data, (1, len(data))).flatten())

        for i in range(0, len(data)):
            data[i] = data[i] * 0.6 + 0.2
        return data

    #not completed
    def un_normalize(self):
        self.values = numpy.reshape(numpy.array(self.values), (len(self.values), 1))
        #self.scaler.inverse_transform(self.values)
        self.values = list(numpy.reshape(self.values, (1, len(self.values))).flatten())

    def get_dates(self):
        return self.dates

    def get_values(self):
        return self.values

    def get_changes(self):
        return self.changes

    def get_changes2(self):
        return self.changes2