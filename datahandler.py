import csv, numpy, os.path, ystockquote
from matplotlib import dates


class DataHandler():

    #Financial Data
    filename = ''
    ticker = ''
    startdate = ''
    enddate = ''
    mean = 0
    max = 0
    data = []

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

    #data normalization functions {-1:1}
    def normalize(self):
        self.mean = numpy.mean(list(float(d[-1]) for d in self.data))
        self.max = numpy.max(list(numpy.abs(float(d[-1])-self.mean) for d in self.data))
        for i,val in enumerate(self.data):
            self.data[i][-1] = (float(val[-1])-self.mean)/self.max

    def un_normalize(self):
        for i,val in enumerate(self.data):
            self.data[i] = (float(val[-1])*self.max)+self.mean

    def get_dates(self):
        return list((dates.datestr2num(s[0]) for s in self.data))

    def get_values(self):
        return list((float(s[-1]) for s in self.data))