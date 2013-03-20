import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp

#Input Data
DAYS = 10
TRAINING = 2600
TESTING = 200
startdate = '20020101'  # YYYYMMDD

#Neural Network
INPUT = DAYS
HIDDEN = 20
OUTPUT = 1

#Training
ITERATIONS = 20
LRATE = 0.8
MOMENTUM = 0.2


#S&P 500
sp500 = dh.DataHandler()
sp500.load_index("%5EGSPC", startdate)

sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT)
sp_net.create_training_data(sp500.value_series(), TRAINING)
train_errors, val_errors = sp_net.train(LRATE, MOMENTUM, ITERATIONS)

out_ser = sp_net.get_output(TRAINING, TESTING)
pp.figure(0)
sp_net.series.plot()
out_ser.plot(color='r')
pp.show(0)

pp.figure(1)
pp.plot(train_errors)
pp.plot(val_errors)
pp.show(1)

#NASDAQ COMPOSITE INDEX
#nasdaq = dh.DataHandler()
#nasdaq.load_index("%5EIXIC", startdate)

#MAJOR MARKET INDEX
#tot = dh.DataHandler()
#tot.load_index("%5EXMI", startdate)

#NYSE COMPOSITE INDEX
#nyse = dh.DataHandler()
#nyse.load_index("%5ENYA", startdate)