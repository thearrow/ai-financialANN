import matplotlib
matplotlib.use("Qt4Agg")
import datahandler as dh
import nethandler as nh
from matplotlib import pyplot as pp

#Input Data
TRAINING_PERCENT = 0.80
DATA_THRESHOLD = 0.002
startdate = '20020101'  # YYYYMMDD
indices = ["%5EGSPC", "%5EIXIC", "%5EFVX", "%5EXMI", "%5ENYA"]

#Neural Network
INPUT = len(indices) * 5
HIDDEN = 10
OUTPUT = 1

#Training
ITERATIONS = 10
LRATE = 0.5
MOMENTUM = 0.6


data = dh.DataHandler()
data.load_indices(indices, startdate, DATA_THRESHOLD)

sp_net = nh.NetHandler(INPUT, HIDDEN, OUTPUT)
sp_net.create_training_data(data, TRAINING_PERCENT)
train_errors, val_errors = sp_net.train(LRATE, MOMENTUM, ITERATIONS)

out_ser = sp_net.get_output(TRAINING_PERCENT)

print sp_net.change_tomorrow()

# correct = 0
# for index, row in out_ser.iteritems():
#     actual = data.data[index]
#     if row > 0 and actual > 0:
#         correct += 1
#     elif row < 0 and actual < 0:
#         correct += 1
# print float(correct)/float(len(data.data)),"% Directional Accuracy"

pp.figure(0)
data.data.ix[:, 0].plot(style='bo-', alpha='0.6')
data.data.ix[:, 5].plot(style='g-', alpha='0.3')
data.data.ix[:, 10].plot(style='y-', alpha='0.3')
data.data.ix[:, 15].plot(style='m-', alpha='0.3')
data.data.ix[:, 20].plot(style='c-', alpha='0.3')
out_ser.plot(style='ro-')
pp.axhline(0, color='black')

pp.figure(1)
pp.plot(train_errors)
pp.plot(val_errors)
pp.show()