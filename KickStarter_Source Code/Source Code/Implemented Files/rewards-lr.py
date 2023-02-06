# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:19:11 2020

@author: VyomeshS
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df_rewards= pd.read_csv('kickstarter_content.csv')
df_rewards.columns = ['Updates', 'Comments', 'Rewards', 'Goal','Pledged', 'Backers',  'Duration in Days',
       'Facebook Friends', 'Facebook Shares', 'Creator - # Projects Created',
       'Creator - # Projects Backed', '# Videos', '# Images',
       '# Words (Description)', '# Words (Risks and Challenges)', '# FAQs',
       'Currency', 'Top Category', 'Category', 'Facebook Connected',
       'Has Video', 'Creator Website', 'State']
df_rewards = df_rewards.set_index("State")
df_rewards = df_rewards.drop([0], axis=0)
df_rewards = df_rewards.reset_index()

a = df_rewards['Goal'].values
b = df_rewards['Rewards'].values
c = df_rewards['Duration in Days'].values
d = df_rewards['Top Category'].values
e = df_rewards['Category'].values


X = np.array([a, b, c, d, e])
X = X.transpose()

y = np.array([df_rewards['Backers'].values])
y = y.transpose()
y = y.ravel()

from sklearn import linear_model, metrics 

# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X, y) 
  
# reg.predict([[df_final.iloc[6][0],
#               df_final.iloc[6][1],
#               df_final.iloc[6][2],
#               df_final.iloc[6][3],
#               df_final.iloc[6][4],
#               df_final.iloc[6][6]]])

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train the model using the training sets 
reg.fit(X_train, y_train) 

# variance score: 1 means perfect prediction 
print('Accuracy: {}'.format(reg.score(X_test, y_test)))
# variance score: 1 means perfect prediction 
print('Accuracy: {}'.format(reg.score(X_test, y_test)))
# Saving model to disk
#pickle.dump(reg, open('back-rewards.pkl','wb'))
b_content=pickle.load(open('back-rewards.pkl', 'rb'))
a_content=pickle.load(open('ap-rewards.pkl', 'rb'))
print(b_content.predict([[6500,7,60,0,0]]))

print(a_content.predict([[6500,7,60,0,0]]))