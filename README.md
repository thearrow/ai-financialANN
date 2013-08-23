Recurrent ANN for Financial Forecasting
======

- Pulls financial data (historical prices of various indices) from the Yahoo Finance API
- Constructs a recurrent neural network with an architecture suitable for time-series forecasting
- Trains the network on that data
- Displays graphs and output data detailing the results


Output
------

Output graph of %-change in price over time of various indices, with prediction for the S&P500 in red, actual S&P500 in blue:
![Output Graph](http://i.imgur.com/sA2g2P7.png "Output Graph")

Output graph of neural network training and validation errors over training epochs:
![Error Graph](http://i.imgur.com/xJExb03.png "Error Graph")

Example output text:
```
train-errors: [  0.043748  0.041089  0.037072  0.036456  0.035725  0.034317  0.033760  0.032625  0.032160  0.032116  0.031686  0.031163  0.030988  0.030825  0.029538  0.029586  0.029203  0.028406  0.028104  0.027824  0.027243  0.030032]
valid-errors: [  0.117359  0.045864  0.042122  0.041731  0.043724  0.039824  0.039880  0.038656  0.039171  0.041646  0.043562  0.035770  0.039031  0.039117  0.040060  0.043550  0.039386  0.039182  0.043095  0.043065  0.041201  0.044457]
Net Topology: 24-12-1

On Fri, Aug 23, 2013 the market will increase.

53.397% Directional Accuracy
```
