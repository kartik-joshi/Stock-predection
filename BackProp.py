#Program for backpropogation for  6 class Dermatology dataset

from math import exp
from random import random
from csv import reader
import numpy as np
import time

print('program execution start')

# Read datasets from CSV input file
def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r', encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string columns to float in input dataset
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer in input dataset (last column with class value)
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# testdataset
testdataset= Read_file('Input_Dataset.csv')
# dataset = Read_file('Input_Dataset.csv')
# x = int(2*len(dataset)/3)
# traindataset = dataset[0:x]
# testdataset = dataset[x:len(dataset)]




# #
# for i in range(len(traindataset[0]) - 1):
#     str_column_to_int(traindataset, i)
# # # convert last column to integers
# str_column_to_int(traindataset, len(traindataset[0]) - 1)
#
# #normalize dataset to get better result
# minmax = dataset_minmax(traindataset)
# normalize_dataset(traindataset, minmax)

for i in range(len(testdataset[0]) - 1):
    str_column_to_int(testdataset, i)
# # convert last column to integers
str_column_to_int(testdataset, len(testdataset[0]) - 1)

#normalize dataset to get better result
minmax = dataset_minmax(testdataset)
normalize_dataset(testdataset, minmax)



# Initialize a network
def init_nw(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate,difference):
    if(difference >= 1):
        new_l_rate = l_rate + (0.1 * difference)
    else: new_l_rate = l_rate
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += new_l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += new_l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            difference = 0
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate,difference)



# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

start_time = time.time()
n_hidden = 6
n_inputs = len(testdataset[0]) - 1
n_outputs = 2
learning_rate = 0.5
n_epoch = 30
w = 0
r = 1
print('Backprop aglo')
runs = [0]*30
for i in range(len(runs)):
    # # learning_rate = learning_rate + 0.05
    # # n_epoch = n_epoch + 4
    r =  r + 1000
    if (r + 10000) > 15800:
        r = w
        w += 300
    traindataset = [testdataset[i] for i in range(r, r + 10000)]
    network1 = init_nw(n_inputs, n_hidden, n_outputs)
    train_network(network1, traindataset, learning_rate, n_epoch, n_outputs)
    total = 0
    misclassification = 0
    total_missclassification_cost = 0.0
    for row in testdataset:
        total = total + 1
        prediction = predict(network1, row)
        if (row[-1] != prediction):
            misclassification = misclassification + 1
    Accuracy =(total - misclassification)*100/total
    print('Accuracy:=%f' %(Accuracy))
    runs[i] = Accuracy

mean = sum(runs)/len(runs)
print("n_epoch: {}".format(n_epoch))
print("learning rate: {}".format(learning_rate))
print("Mean_Accuracy: {}".format(mean))
print("Standard_Deviation: {}".format(np.std(runs, 0)))
time_taken = time.time() - start_time
print("total_time : {}".format(time_taken))

