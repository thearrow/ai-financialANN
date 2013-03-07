import itertools, numpy, datahandler as dh
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit
from pybrain.tools.validation import Validator
from matplotlib import pyplot as pp

#Input Data
INDICES = 4
DAYS = 4
startdate = '20020101'  # YYYYMMDD
enddate = '20130301'  # YYYYMMDD

#Neural Network
INPUT = INDICES * DAYS + INDICES * 2
HIDDEN = 30  # divisible by 4 for net1
OUTPUT = 1

#Training
ITERATIONS = 20
TRAINING = 2200
TESTING = 610
LRATE = 0.8
MOMENTUM = 0.4


#Neural Net Functions
def train(net, data):
    trainer = BackpropTrainer(net, data, learningrate=LRATE, momentum=MOMENTUM, weightdecay=0.0001)
    print "Training..."
    #for _ in range(ITERATIONS):
    #    print trainer.train()
    return trainer.trainUntilConvergence(maxEpochs=ITERATIONS)


def create_training_data(input):
    data = SupervisedDataSet(INPUT, OUTPUT)
    count = 0
    while count + DAYS + OUTPUT < TRAINING:
        ins = []
        for index in input['prices']:
            ins.extend(index[count:count + DAYS])
        for index in input['changes']:
            ins.extend(index[count:count + 1])
        for index in input['changes2']:
            ins.extend(index[count:count + 1])
        outs = input['prices'][0][count + DAYS + 1:count + DAYS + OUTPUT + 1]
        data.addSample(ins, outs)
        count += 1
    return data


def get_output_vals(net, input):
    outputs = []
    count = TRAINING
    while count + DAYS + OUTPUT < TRAINING + TESTING:
        ins = []
        for index in input['prices']:
            ins.extend(index[count:count + DAYS])
        for index in input['changes']:
            ins.extend(index[count:count + 1])
        for index in input['changes2']:
            ins.extend(index[count:count + 1])
        outputs.append(net.activate(ins))
        count += OUTPUT
    realouts = []
    for d in outputs:
        realouts.append(list(d))
    return list(itertools.chain(*realouts))


def get_output_dates(dates):
    return dates[TRAINING + DAYS:TRAINING + TESTING - OUTPUT]


def absolute_errors(output, target):
    errors = []
    for i in range(0, len(output)):
        errors.append((abs(target[i] - output[i])) / target[i])
    return errors


#Main program

def assembleNetwork1():
    # (INPUT : HIDDENSPLIT : HIDDENSPLIT : OUTPUT)
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(INPUT, name="in"))
    n.addModule(SigmoidLayer(HIDDEN / 4, name="h1"))
    n.addModule(SigmoidLayer(HIDDEN / 4, name="h2"))
    n.addModule(LinearLayer(HIDDEN / 4, name="h3"))
    n.addModule(LinearLayer(HIDDEN / 4, name="h4"))
    n.addModule(BiasUnit(name="bias"))
    n.addOutputModule(SigmoidLayer(OUTPUT, name="out"))

    n.addConnection(FullConnection(n['in'], n['h1']))
    n.addConnection(FullConnection(n['in'], n['h3']))
    n.addConnection(FullConnection(n['h1'], n['h2']))
    n.addConnection(FullConnection(n['h3'], n['h4']))
    n.addConnection(FullConnection(n['h4'], n['out']))
    n.addConnection(FullConnection(n['h2'], n['out']))
    n.addConnection(FullConnection(n['bias'], n['out']))
    n.sortModules()
    n = n.convertToFastNetwork()
    return n


def assembleNetwork2():
    # (INPUT : HIDDEN : HIDDEN : OUTPUT)
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(INPUT, name="in"))
    n.addModule(SigmoidLayer(HIDDEN, name="h1"))
    n.addModule(SigmoidLayer(HIDDEN, name="h2"))
    n.addModule(BiasUnit(name="bias"))
    n.addOutputModule(SigmoidLayer(OUTPUT, name="out"))

    n.addConnection(FullConnection(n['bias'], n['in']))
    n.addConnection(FullConnection(n['in'], n['h1']))
    n.addConnection(FullConnection(n['in'], n['h2']))
    n.addConnection(FullConnection(n['h2'], n['out']))
    n.sortModules()
    n = n.convertToFastNetwork()
    return n

#Feed-Forward:
net = assembleNetwork2()
net.randomize()

#S&P 500
sp500 = dh.DataHandler("%5EGSPC", startdate, enddate)
spdates = sp500.get_dates()
spvalues = sp500.get_values()
spchanges = sp500.get_changes()
spchanges2 = sp500.get_changes2()

#NASDAQ COMPOSITE INDEX
nasdaq = dh.DataHandler("%5EIXIC", startdate, enddate)
nasvals = nasdaq.get_values()
naschanges = nasdaq.get_changes()
naschanges2 = nasdaq.get_changes2()

#MAJOR MARKET INDEX
tot = dh.DataHandler("%5EXMI", startdate, enddate)
totvals = tot.get_values()
totchanges = tot.get_changes()
totchanges2 = tot.get_changes2()

#NYSE COMPOSITE INDEX
nyse = dh.DataHandler("%5ENYA", startdate, enddate)
nysevals = nyse.get_values()
nysechanges = nyse.get_changes()
nysechanges2 = nyse.get_changes2()

vals = [spvalues, nasvals, totvals, nysevals]
changes = [spchanges, naschanges, totchanges, nysechanges]
changes2 = [spchanges2, naschanges2, totchanges2, nysechanges2]
aggregated_input = {'prices': vals, 'changes': changes, 'changes2': changes2}
data = create_training_data(aggregated_input)

errors = train(net, data)

sp_predicted = get_output_vals(net, aggregated_input)

val = Validator()
pred_errors = absolute_errors(sp_predicted, spvalues[TRAINING + DAYS + OUTPUT:TRAINING + TESTING])
mse = val.MSE(sp_predicted, spvalues[TRAINING + DAYS + OUTPUT:TRAINING + TESTING])
mape = (100.0 / len(pred_errors)) * numpy.sum(pred_errors)

#Configure plots
ENDTRAINING = TRAINING + DAYS
predicted_dates = get_output_dates(spdates)
pp.subplot2grid((3, 4), (0, 0), colspan=4)
pp.plot_date(spdates, spvalues, linestyle='solid', c='black', marker='None', label="S&P500 Actual")
pp.plot_date(spdates, nasvals, linestyle='dashed', c='g', marker='None', alpha=0.4)
pp.plot_date(spdates, totvals, linestyle='dashed', c='b', marker='None', alpha=0.4)
pp.plot_date(spdates, nysevals, linestyle='dashed', c='m', marker='None', alpha=0.4)
pp.plot_date(predicted_dates, sp_predicted, linestyle='solid', c='r', marker='None', label="S&P500 Predicted")
pp.vlines(spdates[ENDTRAINING], numpy.min(spvalues), numpy.max(spvalues))
pp.ylabel("Normalized Indices")
pp.text(spdates[ENDTRAINING - 350], numpy.max(spvalues) * .9, 'TRAINING')
pp.text(spdates[ENDTRAINING + 50], numpy.max(spvalues) * .9, 'PREDICTION')
pp.legend(fontsize='small', loc=4)
pp.grid(True)

pp.subplot2grid((3, 4), (1, 0), colspan=4)
pp.plot_date(spdates, spchanges, linestyle='none', c='black', marker='.', ms=2.0, label="1-Day Change", alpha=0.75)
pp.plot_date(spdates, spchanges2, linestyle='none', c='green', marker='.', ms=2.0, label="2-Day Change", alpha=0.75)
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
pp.plot_date(predicted_dates, pred_errors, linestyle='solid', c='red', marker='None')
pp.ylabel("Prediction Error", fontsize='small')
errortext = 'Total MSE: %.4f\nTotal MAPE: %.2f' % (mse, mape) + "%"
pp.text(predicted_dates[-1] - 250, numpy.max(pred_errors) * .8, errortext, bbox=dict(color='white', ec='black'))
pp.grid(True)

pp.show()