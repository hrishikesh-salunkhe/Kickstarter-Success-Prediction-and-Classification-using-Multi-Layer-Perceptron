from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np




# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
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
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])




# Test Backprop on Seeds dataset
seed(1)

# load and prepare data
filename = 'kickstarter_general_final.csv'



dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
 

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)


# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

class MLP:

  n_folds = 3
  l_rate = 0.3
  n_epoch = 1000
  n_hidden_1 = 8
  network = list()

  def initialize_network(self, n_inputs, n_outputs):
      network = list()
      hidden_layer_1 = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(self.n_hidden_1)]
      network.append(hidden_layer_1)
      output_layer = [{'weights':[random() for i in range(self.n_hidden_1 + 1)]} for i in range(n_outputs)]
      network.append(output_layer)
      return network

  def cross_validation_split(self, dataset):
      dataset_split = list()
      dataset_copy = list(dataset)
      fold_size = int(len(dataset) / self.n_folds)
      for i in range(self.n_folds):
        fold = list()
        while len(fold) < fold_size:
          index = randrange(len(dataset_copy))
          fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
      return dataset_split

  def accuracy_metric(self, actual, predicted):
      correct = 0
      for i in range(len(actual)):
        if actual[i] == predicted[i]:
          correct += 1
      return correct / float(len(actual)) * 100.0

  def evaluate_algorithm(self, dataset):
      folds = self.cross_validation_split(dataset)
      fold_count = 0
      scores = list()
      for fold in folds:
        fold_count = fold_count + 1
        print('Fold Number: ', fold_count)
        print(' ')
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
          row_copy = list(row)
          test_set.append(row_copy)
          row_copy[-1] = None
        predicted = self.back_propagation(train_set, test_set)
        actual = [row[-1] for row in fold]
        accuracy = self.accuracy_metric(actual, predicted)
        scores.append(accuracy)
      return scores

  def activate(self, weights, inputs):
      activation = weights[-1]
      for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
      return activation

  def transfer(self, activation):
		  return 1.0 / (1.0 + exp(-activation))
    
  def forward_propagate(self, network, row):
      inputs = row
      for layer in network:
        new_inputs = []
        for neuron in layer:
          activation = self.activate(neuron['weights'], inputs)
          neuron['output'] = self.transfer(activation)
          new_inputs.append(neuron['output'])
        inputs = new_inputs
      return inputs

  def transfer_derivative(self, output):
		  return output * (1.0 - output)
    
  def backward_propagate_error(self, network, expected):
      for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
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
          neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

  def update_weights(self, network, row):
      for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
          inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
          for j in range(len(inputs)):
            neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
          neuron['weights'][-1] += self.l_rate * neuron['delta']

  def train_network(self, network, train, n_outputs):
      for epoch in range(self.n_epoch):
        print('Epoch: ',epoch)
        for row in train:
          outputs = self.forward_propagate(network, row)
          expected = [0 for i in range(n_outputs)]
          expected[row[-1]] = 1
          self.backward_propagate_error(network, expected)
          self.update_weights(network, row)
      return network

  def back_propagation(self, train, test):
      n_inputs = len(train[0]) - 1
      n_outputs = len(set([row[-1] for row in train]))
      network = self.initialize_network(n_inputs, n_outputs)
      self.network.append(self.train_network(network, train, n_outputs))
      network = self.network[-1]
      predictions = list()
      for row in test:
        prediction = self.predict(network, row)
        predictions.append(prediction)
      return(predictions)

  def predict(self, network, row):
      outputs = self.forward_propagate(network, row)
      return outputs.index(max(outputs))
  
    
    
network = np.load('general_network.npy', allow_pickle = True)
network = network.tolist()

mlp = MLP()
input=[5000	,10,	5,	0,	0,	28]
input.append(1)
data=list()
data.append(input)


#np.save('minmax_general.npy', minmax)
normalize_dataset(data,minmax)
data[0].pop()
print(mlp.predict(network[0], data[0]))
