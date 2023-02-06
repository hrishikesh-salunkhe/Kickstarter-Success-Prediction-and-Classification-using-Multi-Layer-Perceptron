# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:02:50 2020

@author: VyomeshS
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df_general = pd.read_csv('kickstarter_general.csv')
df_general.columns = ['goal', 'pledged', 'backers', 'category', 'main category','country', 'currency', 'duration', 'state']

df_general = df_general.set_index("state")
df_general = df_general.drop([0], axis=0)
df_general = df_general.reset_index()

a = df_general['goal'].values
b = df_general['category'].values
c = df_general['main category'].values
d = df_general['country'].values
e = df_general['currency'].values
f = df_general['duration'].values

X = np.array([a, b, c, d, e, f])
X = X.transpose()

y = np.array([df_general['backers'].values])
y = y.transpose()
y = y.ravel()

from sklearn import linear_model, metrics 

# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X, y) 
 
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train the model using the training sets 
reg.fit(X_train, y_train) 

# variance score: 1 means perfect prediction 
print('Accuracy: {}'.format(reg.score(X_test, y_test))) 
  
 



# Saving model to disk
pickle.dump(reg, open('back-general.pkl','wb'))

# Loading model to compare the results
ap_general = pickle.load(open('back-general.pkl','rb'))
print(ap_general.predict([[6000,0,0,0,0,30]]))