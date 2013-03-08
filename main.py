import numpy, datahandler as dh, trainerhandler as th
from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit
from pybrain.tools.validation import Validator
from matplotlib import pyplot as pp

#Input Data
INDICES = 4
CHANGE_DAYS = 6
startdate = '20020101'  # YYYYMMDD

#Training
ITERATIONS = 20
TRAIN = 2400
TEST = 200
LRATE = 0.8
MOMENTUM = 0.5

#Neural Network
INPUT = INDICES * CHANGE_DAYS
HIDDEN = 30  # divisible by 4 for net1
OUTPUT = 1


def absolute_errors(output, target):
    errors = []
    for i in range(0, len(output)):
        errors.append((abs(target[i] - output[i])) / target[i])
    return errors


#Main program


def assembleNetwork():
    # (INPUT : HIDDEN : HIDDEN : OUTPUT)
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(INPUT, name="in"))
    n.addModule(SigmoidLayer(HIDDEN, name="h1"))
    n.addModule(SigmoidLayer(HIDDEN, name="h2"))
    n.addModule(BiasUnit(name="bias"))
    n.addOutputModule(SigmoidLayer(OUTPUT, name="out"))

    n.addConnection(FullConnection(n['bias'], n['in']))
    n.addConnection(FullConnection(n['in'], n['h1']))
    n.addConnection(FullConnection(n['h1'], n['h2']))
    n.addConnection(FullConnection(n['h2'], n['out']))
    n.sortModules()
    n = n.convertToFastNetwork()
    return n

#Feed-Forward:
net = assembleNetwork()
net.randomize()

#S&P 500
sp500 = dh.DataHandler()
sp500.load_index("%5EGSPC", startdate, CHANGE_DAYS)
spdates = sp500.get_dates()
spvalues = sp500.get_values()
spchanges = sp500.get_changes()

#NASDAQ COMPOSITE INDEX
nasdaq = dh.DataHandler()
nasdaq.load_index("%5EIXIC", startdate, CHANGE_DAYS)
nasvals = nasdaq.get_values()
naschanges = nasdaq.get_changes()

#MAJOR MARKET INDEX
tot = dh.DataHandler()
tot.load_index("%5EXMI", startdate, CHANGE_DAYS)
totvals = tot.get_values()
totchanges = tot.get_changes()

#NYSE COMPOSITE INDEX
nyse = dh.DataHandler()
nyse.load_index("%5ENYA", startdate, CHANGE_DAYS)
nysevals = nyse.get_values()
nysechanges = nyse.get_changes()

trainer = th.TrainerHandler(INDICES, CHANGE_DAYS, INPUT, OUTPUT, ITERATIONS, TRAIN, TEST, LRATE, MOMENTUM)
datahandlers = [sp500, nasdaq, tot, nyse]
trainer.create_data(datahandlers, 0)
errors = trainer.perform_training(net)
sp_predicted = trainer.get_output_vals(net, datahandlers)

#sp_predicted = sp500.normalize(sp_predicted, sp500.get_scalers()[0], 0)
predicted_dates = trainer.get_output_dates(sp500.get_dates())

# val = Validator()
# pred_errors = absolute_errors(sp_predicted, spchanges[0][TRAIN + CHANGE_DAYS + OUTPUT:TRAIN + TEST])
# mse = val.MSE(sp_predicted, spchanges[0][TRAIN + CHANGE_DAYS + OUTPUT:TRAIN + TEST])
# mape = (100.0 / len(pred_errors)) * numpy.sum(pred_errors)


#Configure plots
ENDTRAINING = TRAIN + CHANGE_DAYS + OUTPUT
pp.subplot2grid((3, 4), (0, 0), colspan=4)
pp.plot_date(spdates, spvalues, linestyle='solid', c='black', marker='None', label="S&P500")
# pp.plot_date(spdates, nasvals, linestyle='dashed', c='g', marker='None', alpha=0.4)
# pp.plot_date(spdates, totvals, linestyle='dashed', c='b', marker='None', alpha=0.4)
# pp.plot_date(spdates, nysevals, linestyle='dashed', c='m', marker='None', alpha=0.4)
pp.vlines(spdates[ENDTRAINING], numpy.min(spvalues), numpy.max(spvalues))
pp.ylabel("Normalized Indices")
pp.text(spdates[ENDTRAINING - 350], numpy.max(spvalues) * .9, 'TRAINING')
pp.text(spdates[ENDTRAINING + 50], numpy.max(spvalues) * .9, 'PREDICTION')
pp.legend(fontsize='small', loc=4)
pp.grid(True)

pp.subplot2grid((3, 4), (1, 0), colspan=4)
pp.plot_date(predicted_dates, spchanges[0][TRAIN + CHANGE_DAYS:TRAIN + TEST - OUTPUT], linestyle='none',
             c='black', marker='.', ms=3.0, label="Actual 1-Day Change", alpha=0.75)
pp.plot_date(predicted_dates, sp_predicted, linestyle='none', c='red', marker='.', ms=3.0, label="Predicted Change")
pp.ylabel("Normalized Change")
pp.grid(True)
pp.legend(fontsize='small', loc=4)

pp.subplot2grid((3, 4), (2, 0))
pp.plot(numpy.reshape(errors[0], (len(errors[0]), 1)), c='r')
pp.xlabel("Training Epoch Number")
pp.ylabel("Mean Squared Error", fontsize='small')
pp.grid(True)

pp.subplot2grid((3, 4), (2, 1))
pp.plot(numpy.reshape(errors[1], (len(errors[1]), 1)), c='r')
pp.xlabel("Validation Epoch Number")
pp.grid(True)

pp.subplot2grid((3, 4), (2, 2), colspan=2)
# pp.plot_date(predicted_dates, pred_errors, linestyle='solid', c='red', marker='None')
pp.ylabel("Prediction Error", fontsize='small')
# errortext = 'Total MSE: %.4f\nTotal MAPE: %.2f' % (mse, mape) + "%"
# pp.text(predicted_dates[-1] - 250, numpy.max(pred_errors) * .8, errortext, bbox=dict(color='white', ec='black'))
pp.grid(True)

pp.show()