import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import syllables
import textstat
import readability


app = Flask(__name__)


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
        
    return unique_list

def wiener_sachtext_formel(ms, sl, iw, es):
  wsf = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
  return wsf

class MLP_rewards:

  n_folds = 3
  l_rate = 0.3
  n_epoch = 1000
  n_hidden_1 = 12
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

class MLP_content:

  
    n_folds = 3
    l_rate = 0.3
    n_epoch = 1000
    n_hidden_1 = 22
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
  


class MLP_general:

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
  


class MLP_description:

    n_folds = 3
    l_rate = 0.3
    n_epoch = 1000
    n_hidden_1 = 24
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
  
 














#1 Loading Linear Regression Pickle Files 
#Amount Pledge      
ap_general = pickle.load(open('ap-general.pkl', 'rb'))
ap_content=pickle.load(open('ap-content.pkl', 'rb'))
ap_desc=pickle.load(open('ap-desc.pkl', 'rb'))
ap_rewards=pickle.load(open('ap-rewards.pkl', 'rb'))


#Backers
b_content=pickle.load(open('back-content.pkl', 'rb'))
b_general=pickle.load(open('back-general.pkl', 'rb'))
b_desc=pickle.load(open('back-desc.pkl', 'rb'))
b_rewards=pickle.load(open('back-rewards.pkl', 'rb'))


#2 Loading Minmax Files

minmax_rewards=np.load('minmax_rewards.npy')
minmax_rewards = minmax_rewards.tolist()

minmax_content=np.load('minmax_content.npy')
minmax_content = minmax_content.tolist()

minmax_general=np.load('minmax_general.npy')
minmax_general = minmax_general.tolist()

minmax_description=np.load('minmax_description.npy')
minmax_description = minmax_description.tolist()

#3  Loading MLP Network Files
rewards_network=np.load('rewards_network.npy', allow_pickle = True)
rewards_network = rewards_network.tolist()

content_network=np.load('content_network.npy', allow_pickle = True)
content_network= content_network.tolist()

general_network=np.load('general_network.npy', allow_pickle = True)
general_network= general_network.tolist()

description_network=np.load('description_network.npy', allow_pickle = True)
description_network= description_network.tolist()




#Rendering HTML files based on route selected

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/apg')
def apg():
     return render_template('ap-general.html')

@app.route('/apc')
def apc():
     return render_template('ap-content.html')


@app.route('/apr')
def apr():
     return render_template('ap-reward.html')
@app.route('/apd')
def apd():
    return render_template('ap-desc.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/catstats')
def catstats():
    return render_template('catstats.html')

@app.route('/rewards_mlp')
def rewards_mlp():
    return render_template('rewards_mlp.html')


@app.route('/content_mlp')
def content_mlp():
    return render_template('content_mlp.html')


@app.route('/general_mlp')
def general_mlp():
    return render_template('general_mlp.html')


@app.route('/description_mlp')
def description_mlp():
    return render_template('description_mlp.html')





# Linear Regression Processing Starts From Here
#General Dataset    
@app.route('/apg_predict',methods=['POST'])
def apg_predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    #Estimating Amount Pledge
    prediction = ap_general.predict(final_features)
    
    output = round(prediction[0], 2)
    #Estimating Backers
    prediction1 = b_general.predict(final_features)
    
    output1 = round(prediction1[0], 2)
    return render_template('ap-general.html', prediction_text='Estimated Amount Pledge: {} with Approximate Backers: {}'.format(output,output1))

#Description Dataset
@app.route('/apd_predict',methods=['POST'])
def apd_predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model1.predict(final_features)
    
    
    desc = request.form['description']
    goal = request.form['goal']  
    category = request.form['category']  
    main_category = request.form['maincategory']  
    country = request.form['country']  
    currency = request.form['currency']  
    duration = request.form['duration']
    goal=int(goal)
    category=int(category)
    main_category=int(main_category)
    country=int(country)
    currency=int(currency)
    duration=int(duration)
    words_list = unique(desc.split())
    n_words = len(words_list)
    
    
    n_sents = desc.count('.')
    if(n_sents==0):
        n_sents=1;
    n_chars = len(desc) - desc.count(' ') - desc.count('.')
    n_syllables = 0
    n_monosyllable_words = 0
    n_polysyllable_words = 0
    
    
    for i in range(0,len(words_list)):
        n_syllables += syllables.estimate(words_list[i])
        if(syllables.estimate(words_list[i]) == 1):
            n_monosyllable_words += 1
        elif(syllables.estimate(words_list[i]) > 2):
            n_polysyllable_words +=1
    n_syllables = int(n_syllables)
    n_monosyllable_words = int(n_monosyllable_words)
    n_unique_words = len(unique(words_list))   
    n_long_words = 0
    for i in range(0,len(words_list)):
        if(len(words_list[i]) > (n_chars/n_words)):
                n_long_words = n_long_words + 1
    #n_long_words=int(n_long_words)
    flesch_kincaid_grade_level = int(textstat.flesch_kincaid_grade(desc))
    flesch_reading_ease = int(textstat.flesch_reading_ease(desc))
    smog_index = int(textstat.smog_index(desc))
    gunning_fog_index = int(textstat.gunning_fog(desc))
    coleman_liau_index = int(textstat.coleman_liau_index(desc))
    automated_readability_index = int(textstat.automated_readability_index(desc))
    results = readability.getmeasures(desc, lang='en')
    lix = int(results['readability grades']['LIX'])
    avg_sentence_length = int(n_words / n_sents)
    number_of_words_with_six_or_more_characters = 0

    for i in range(0, len(words_list)):
        if(len(words_list[i]) > 5):
            number_of_words_with_six_or_more_characters += 1

    wsf = wiener_sachtext_formel(
		n_polysyllable_words / n_words * 100,
		avg_sentence_length,
		number_of_words_with_six_or_more_characters / n_words * 100,
		n_monosyllable_words / n_words * 100)

    wsf=int(wsf)
    data=list()
    data.append([n_words,n_sents,n_chars,n_syllables,n_unique_words,n_long_words,n_monosyllable_words,n_polysyllable_words,flesch_kincaid_grade_level,flesch_reading_ease,smog_index,gunning_fog_index,coleman_liau_index,automated_readability_index,lix,wsf,goal,category,main_category,country,currency,duration] )
    for i in range(len(data[0])):
        data[0][i]=float(data[0][i])
    #Estimating Amount Pledge
    prediction = ap_desc.predict(data)
    output = round(prediction[0], 2)
    #Estimating Backers
    prediction1 = b_desc.predict(data)
    output1 = round(prediction1[0], 2)
    
    return render_template('ap-desc.html', prediction_text='Estimated Amount Pledge {} with Approximate Backers: {}'.format(output,output1))

@app.route('/apc_predict',methods=['POST'])
def apc_predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #Estimating Amount Pledge
    prediction = ap_content.predict(final_features)
    output = round(prediction[0], 2)

    #Estimating Backers
    prediction1 = b_content.predict(final_features)
    output1 = round(prediction1[0], 2)
    
    return render_template('ap-content.html', prediction_text='Estimated Amount Pledge {} with Approximate Backers: {}'.format(output,output1))



@app.route('/apr_predict',methods=['POST'])
def apr_predict():
    '''
    For rendering results on HTML GUI
    '''
    
    goal = request.form['goal']  
    rewards=request.form['levels']  
    duration = request.form['duration']
    main_category = request.form['maincategory']  
    category = request.form['category']  
       
    
    
    data=list()
    data.append([goal,rewards,duration,main_category,category])
    for i in range(len(data[0])):
        data[0][i]=float(data[0][i])
    
    #Estimating Amount Pledge
    
    prediction = ap_rewards.predict(data)
    
    output = round(prediction[0], 2)
    #Estimating Backers
    prediction1 = b_rewards.predict(data)
    output1 = round(prediction1[0], 2)
    
    return render_template('ap-reward.html',prediction_text='Estimated Amount Pledge {} with Approximate Backers: {}'.format(output,output1))

#MLP processing starts from here
@app.route('/rewards',methods=['POST'])
def rewards():
      
    int_features = [int(x) for x in request.form.values()]
    
   
    mlp_rewards = MLP_rewards()
    #input=[6500	,765	,20,	7	,10,	0	,60.34,	7,	11,	5,	300,	87.85714286]
    int_features.append(1)
    data=list()
    data.append(int_features)

   
    normalize_dataset(data,minmax_rewards)
    data[0].pop()
    output=mlp_rewards.predict(rewards_network[0], data[0])
    if output==1:
        result="Project Can Succeed"
    elif output==0:  
        result="Project May Fail"

    return render_template('rewards_mlp.html', prediction_text='Result {}'.format(result))

@app.route('/content',methods=['POST'])
def content():
      
    int_features = [int(x) for x in request.form.values()]
    
    int_features.append([1])
    mlp_content = MLP_content()
    #input=[6500	,765	,20,	7	,10,	0	,60.34,	7,	11,	5,	300,	87.85714286]
    data=list()
    data.append(int_features)
    
   
    normalize_dataset(data,minmax_content)
    data[0].pop()
    output=mlp_content.predict(content_network[0], data[0])
    if output==1:
        result="Project Can Succeed"
    elif output==0:  
        result="Project May Fail"
    return render_template('content_mlp.html', prediction_text='Result {}'.format(result))    

@app.route('/general',methods=['POST'])
def general():
      
    int_features = [int(x) for x in request.form.values()]
    
   
    mlp_general = MLP_general()
    #input=[6500	,765	,20,	7	,10,	0	,60.34,	7,	11,	5,	300,	87.85714286]
    int_features.append(1)
    data=list()
    data.append(int_features)

   
    normalize_dataset(data,minmax_general)
    data[0].pop()
    output=mlp_general.predict(general_network[0], data[0])
    if output==1:
        result="Project Can Succeed"
    elif output==0:  
        result="Project May Fail"
    return render_template('general_mlp.html', prediction_text='Result {}'.format(result)) 

@app.route('/description',methods=['POST'])
def description():
      
    desc = request.form['description']
    goal = request.form['goal']  
    category = request.form['category']  
    main_category = request.form['maincategory']  
    country = request.form['country']  
    currency = request.form['currency']  
    duration = request.form['duration']
    goal=int(goal)
    category=int(category)
    main_category=int(main_category)
    country=int(country)
    currency=int(currency)
    duration=int(duration)
    words_list = unique(desc.split())
    n_words = len(words_list)
    
    
    n_sents = desc.count('.')
    if(n_sents==0):
        n_sents=1;
    n_chars = len(desc) - desc.count(' ') - desc.count('.')
    n_syllables = 0
    n_monosyllable_words = 0
    n_polysyllable_words = 0
    
    
    for i in range(0,len(words_list)):
        n_syllables += syllables.estimate(words_list[i])
        if(syllables.estimate(words_list[i]) == 1):
            n_monosyllable_words += 1
        elif(syllables.estimate(words_list[i]) > 2):
            n_polysyllable_words +=1
    n_syllables = int(n_syllables)
    n_monosyllable_words = int(n_monosyllable_words)
    n_unique_words = len(unique(words_list))   
    n_long_words = 0
    for i in range(0,len(words_list)):
        if(len(words_list[i]) > (n_chars/n_words)):
                n_long_words = n_long_words + 1
    #n_long_words=int(n_long_words)
    flesch_kincaid_grade_level = int(textstat.flesch_kincaid_grade(desc))
    flesch_reading_ease = int(textstat.flesch_reading_ease(desc))
    smog_index = int(textstat.smog_index(desc))
    gunning_fog_index = int(textstat.gunning_fog(desc))
    coleman_liau_index = int(textstat.coleman_liau_index(desc))
    automated_readability_index = int(textstat.automated_readability_index(desc))
    results = readability.getmeasures(desc, lang='en')
    lix = int(results['readability grades']['LIX'])
    avg_sentence_length = int(n_words / n_sents)
    number_of_words_with_six_or_more_characters = 0

    for i in range(0, len(words_list)):
        if(len(words_list[i]) > 5):
            number_of_words_with_six_or_more_characters += 1

    wsf = wiener_sachtext_formel(
		n_polysyllable_words / n_words * 100,
		avg_sentence_length,
		number_of_words_with_six_or_more_characters / n_words * 100,
		n_monosyllable_words / n_words * 100)

    wsf=int(wsf)
    mlp_description = MLP_description()
    #input=[6500	,765	,20,	7	,10,	0	,60.34,	7,	11,	5,	300,	87.85714286]
    data=list()
    data.append([n_words,n_sents,n_chars,n_syllables,n_unique_words,n_long_words,n_monosyllable_words,n_polysyllable_words,flesch_kincaid_grade_level,flesch_reading_ease,smog_index,gunning_fog_index,coleman_liau_index,automated_readability_index,lix,wsf,goal,category,main_category,country,currency,duration,1] )
   
    normalize_dataset(data,minmax_description)
    data[0].pop()
    for i in range(len(data[0])):
        data[0][i]=float(data[0][i])
    output=mlp_description.predict(description_network[0], data[0])

    if output==1:
        result="Project Can Succeed"
    elif output==0:  
        result="Project May Fail"
    return render_template('description_mlp.html', prediction_text='Result {}'.format(result)) 











if __name__ == "__main__":
    app.run(debug=True)