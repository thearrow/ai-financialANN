import csv, numpy, os.path, ystockquote
from matplotlib import dates
from sklearn.preprocessing import MinMaxScaler
from datetime import date


class DataHandler():

    def __init__(self):
        #Financial Data
        self.filename = ''
        self.ticker = ''
        self.startdate = ''
        self.enddate = ''
        self.scalers = []
        self.data = []
        self.dates = []
        self.values = []
        self.changes = []

    #fetch financial data from file or yahoo API
    def load_index(self, ticker, startdate, change_days):
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
        self.values = self.normalize(list(float(s[-1]) for s in self.data),
                                     MinMaxScaler(feature_range=(0, 1), copy=False), 1)
        self.dates = list((dates.datestr2num(s[0]) for s in self.data))
        self.create_changes(change_days)

    def create_changes(self, change_days):
        changes = []
        for i in range(1, change_days + 1):
            self.scalers.append(MinMaxScaler(feature_range=(0, 1), copy=False))
            changes.append(self.normalize(self.get_n_day_change(i), self.scalers[i - 1], 1))
        self.changes = changes

    def get_n_day_change(self, n):
        changes = []
        for i in range(0, n):
            changes.append(0)
        for i in range(n, len(self.values)):
            changes.append(self.values[i] - self.values[i - n])
        return changes

    #data normalization (0.2,0.8)
    def normalize(self, input_data, scaler, fit):
        data = numpy.reshape(numpy.array(input_data), (len(input_data), 1))
        if fit == 1:
            scaler.fit(data)
        scaler.transform(data)
        data = list(numpy.reshape(data, (1, len(data))).flatten())
        for i in range(0, len(data)):
            data[i] = data[i] * 0.6 + 0.2
        return data

    def un_normalize(self, input_data, scaler):
        for i in range(0, len(input_data)):
            input_data[i] = (input_data[i] - 0.2) / 0.6
        data = numpy.reshape(numpy.array(input_data), (len(input_data), 1))
        scaler.inverse_transform(data)
        data = list(numpy.reshape(data, (1, len(data))).flatten())
        return data

    def get_dates(self):
        return self.dates

    def get_values(self):
        return self.values

    def get_changes(self):
        return self.changes

    def get_scalers(self):
        return self.scalers