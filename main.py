import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp

#Input Data
DAYS = 5
TRAINING = 2000
TESTING = 300
startdate = '20020101'  # YYYYMMDD

#Neural Network
INPUT = DAYS
HIDDEN = 20
OUTPUT = 1

#Training
ITERATIONS = 20
LRATE = 0.8
MOMENTUM = 0.01


#S&P 500
sp500 = dh.DataHandler()
sp500.load_index("%5EGSPC", startdate)

sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT)
change_series = []
for i in range(1, DAYS + 1):
    change_series.append(sp500.change_series(i))
sp_net.create_training_data(change_series, TRAINING)
train_errors, val_errors = sp_net.train(LRATE, MOMENTUM, ITERATIONS)

out_ser = sp_net.get_output(TRAINING, TESTING)
pp.figure(0)
sp_net.series[0].plot(style='bo')
out_ser.plot(style='ro')
pp.show(0)

correct = 0
i = 0
for val in sp_net.series[0][out_ser.index]:
    if (val > 0 and out_ser[i] > 0) or (val < 0 and out_ser[i] < 0):
        correct += 1
    i += 1

print float(correct) / float(i)

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