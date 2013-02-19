import csv, numpy, os.path, ystockquote
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from matplotlib import pyplot
from matplotlib import dates

#Financial Data
sp500filename = 'sp500.csv'
nasdaqfilename = 'nasdaq.csv'
startdate = '20100101' #YYYYMMDD
enddate = '20130220' #YYYYMMDD
sp500mean = 0
sp500max = 0
nasdaqmean = 0
nasdaqmax = 0

#Neural Network
INPUT = 15
HIDDEN = 10
OUTPUT = 5
ITERATIONS = 40
TRAINING = 300

#fetch financial data from file or yahoo API
def loadIndex(file):
    if os.path.isfile(file):
        return list(csv.reader(open(file,'rb'),delimiter=','))
    else:
        data = []
        if file == sp500filename:
            data = ystockquote.get_historical_prices("%5EGSPC",startdate,enddate)
        elif file == nasdaqfilename:
            data = ystockquote.get_historical_prices("%5EIXIC",startdate,enddate)
        data.pop(0)
        with open(file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return data

#data normalization functions {-1:1}
def normalize(index, data):
    mean = numpy.mean(list(float(d[-1]) for d in data))
    max = numpy.max(list(numpy.abs(float(d[-1])-mean) for d in data))
    if index == 'sp500':
        global sp500mean
        sp500mean = mean
        global sp500max
        sp500max = max
    elif index == 'nasdaq':
        global nasdaqmean
        nasdaqmean = mean
        global nasdaqmax
        nasdaqmax = max
    for i,val in enumerate(data):
        data[i][-1] = (float(val[-1])-mean)/max

def unNormalize(index, data):
    if index == 'sp500':
        mean = sp500mean
        max = sp500max
    elif index == 'nasdaq':
        mean = nasdaqmean
        max = nasdaqmax
    for i,val in enumerate(data):
        data[i][-1] = (float(val[-1])*max)+mean





sp500 = loadIndex(sp500filename)
normalize('sp500',sp500)
spdates = list((dates.datestr2num(s[0]) for s in sp500))
spvalues = list((s[-1] for s in sp500))

nasdaq = loadIndex(nasdaqfilename)
normalize('nasdaq',nasdaq)
nasdates = list((dates.datestr2num(s[0]) for s in nasdaq))
nasvalues = list((s[-1] for s in nasdaq))



pyplot.plot_date(spdates,spvalues,linestyle='solid',c='b',marker='None')
pyplot.plot_date(nasdates,nasvalues,linestyle='solid',c='g',marker='None')

pyplot.show()