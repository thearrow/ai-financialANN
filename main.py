import matplotlib
#matplotlib.use("MacOSX")
import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp

#Input Data
TRAINING_PERCENT = 0.50
LAG_DAYS = 3
startdate = '20000101'  # YYYYMMDD
indices = ["%5EGSPC", "%5EIXIC", "%5EFVX", "%5ETYX", "%5EXMI", "%5ENYA"]

#Neural Network
INPUT = len(indices) * (LAG_DAYS+1)
HIDDEN = 12
OUTPUT = 1

#Training
ITERATIONS = 20
LRATE = 0.4
MOMENTUM = 0.6


data = dh.DataHandler()
data.load_indices(indices, startdate, LAG_DAYS)
data.create_data(INPUT, OUTPUT)
train, test = data.get_datasets(TRAINING_PERCENT)
print "Training:", len(train), "Testing:", len(test)

sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT, data)
train_errors, val_errors = sp_net.train(train, LRATE, MOMENTUM, ITERATIONS)

out_ser = sp_net.get_output(test, TRAINING_PERCENT)
print "Net Topology: %d-%d-%d" % (INPUT, HIDDEN, OUTPUT)
print sp_net.change_tomorrow()

correct = 0
total = 0
misses = 0
for index, row in out_ser.iteritems():
    try:
        actual = data.dataframe.ix[:, 0][index]
        total += 1
        if row > 0 and actual > 0:
            correct += 1
        elif row < 0 and actual < 0:
            correct += 1
    except KeyError:
        misses += 1
print "%.3f%% Directional Accuracy" % (float(correct) / float(total) * 100)
print "(%d misses)" % misses

pp.figure(0)
data.dataframe.ix[:, 0].plot(style='bo-', alpha=0.8)
data.dataframe.ix[:, (LAG_DAYS+1) * 1].plot(style='g-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 2].plot(style='y-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 3].plot(style='m-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 4].plot(style='c-', alpha=0.5)
data.dataframe.ix[:, (LAG_DAYS+1) * 5].plot(style='-', color='0.75', alpha=0.5)
out_ser.plot(style='ro-')
pp.axhline(0, color='black')

pp.figure(1)
pp.plot(train_errors)
pp.plot(val_errors)
pp.show()