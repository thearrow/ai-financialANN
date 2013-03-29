import matplotlib
matplotlib.use("Qt4Agg")
import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp

#Input Data
TRAINING_PERCENT = 0.80
DATA_THRESHOLD = 0.002
startdate = '20020101'  # YYYYMMDD

#Neural Network
INPUT = 5
HIDDEN = 5
OUTPUT = 1

#Training
ITERATIONS = 20
LRATE = 0.5
MOMENTUM = 0.6

#S&P 500
sp500 = dh.DataHandler()
sp500.load_index("%5EGSPC", startdate, DATA_THRESHOLD)

sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT)
sp_net.create_training_data(sp500, TRAINING_PERCENT)
train_errors, val_errors = sp_net.train(LRATE, MOMENTUM, ITERATIONS)

out_ser = sp_net.get_output(TRAINING_PERCENT)

print sp_net.change_tomorrow()

pp.figure(0)
sp500.data.ix[:, 0].plot(style='bo-')
out_ser.plot(style='ro-')
pp.axhline(0, color='black')

pp.figure(1)
pp.plot(train_errors)
pp.plot(val_errors)
pp.show()


#NASDAQ COMPOSITE INDEX
#nasdaq = dh.DataHandler()
#nasdaq.load_index("%5EIXIC", startdate)

#MAJOR MARKET INDEX
#tot = dh.DataHandler()
#tot.load_index("%5EXMI", startdate)

#NYSE COMPOSITE INDEX
#nyse = dh.DataHandler()
#nyse.load_index("%5ENYA", startdate)