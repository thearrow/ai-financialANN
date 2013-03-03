import csv, numpy, os.path, ystockquote
from matplotlib import dates
from sklearn.preprocessing import MinMaxScaler


class DataHandler():

    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    price_scaler = MinMaxScaler(feature_range=(0,1),copy=False)
    change_scaler = MinMaxScaler(feature_range=(0,1),copy=False)
    change2_scaler = MinMaxScaler(feature_range=(0,1),copy=False)
    data = []
    dates = []
    values = []
    changes = []
    changes2 = []

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
        self.changes = self.createChanges(1)
        self.changes2 = self.createChanges(2)
        self.changes = self.normalize(self.changes, self.change_scaler)
        self.changes2 = self.normalize(self.changes2, self.change2_scaler)
        self.values = self.normalize(self.values, self.price_scaler)

    def createChanges(self,days):
        changes = []
        for i in range(0,days):
            changes.append(0)
        #remaining changes
        for i in range(days,len(self.values)):
             changes.append(self.values[i]-self.values[i-days])
        return changes

    #data normalization (0,1)
    def normalize(self, input, scaler):
        data = numpy.reshape(numpy.array(input),(len(input),1))
        scaler.fit_transform(data)
        return list(numpy.reshape(data,(1,len(data))).flatten())

    #not completed
    def un_normalize(self):
        self.values = numpy.reshape(numpy.array(self.values),(len(self.values),1))
        #self.scaler.inverse_transform(self.values)
        self.values = list(numpy.reshape(self.values,(1,len(self.values))).flatten())

    def get_dates(self):
        return self.dates

    def get_values(self):
        return self.values

    def get_changes(self):
        return self.changes

    def get_changes2(self):
        return self.changes2