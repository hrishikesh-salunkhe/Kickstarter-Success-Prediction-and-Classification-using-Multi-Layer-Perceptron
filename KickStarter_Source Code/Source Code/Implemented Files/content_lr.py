# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:19:11 2020

@author: VyomeshS
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df_content = pd.read_csv('kickstarter_content.csv')



df_content.columns = ['Updates', 'Comments', 'Rewards', 'Goal','Pledged', 'Backers',  'Duration in Days',
       'Facebook Friends', 'Facebook Shares', 'Creator - # Projects Created',
       'Creator - # Projects Backed', '# Videos', '# Images',
       '# Words (Description)', '# Words (Risks and Challenges)', '# FAQs',
       'Currency', 'Top Category', 'Category', 'Facebook Connected',
       'Has Video', 'Creator Website', 'State']


df_content = df_content.set_index("State")
df_content = df_content.drop([0], axis=0)
df_content = df_content.reset_index()

a = df_content['Rewards'].values
b = df_content['Goal'].values
c = df_content['Duration in Days'].values
d = df_content['Facebook Friends'].values
e = df_content['Facebook Shares'].values
f = df_content['Creator - # Projects Created'].values
g = df_content['Creator - # Projects Backed'].values
h = df_content['# Videos'].values
i = df_content['# Images'].values
j = df_content['# Words (Description)'].values
k = df_content['# Words (Risks and Challenges)'].values
l = df_content['# FAQs'].values
m = df_content['Currency'].values
n = df_content['Top Category'].values
o = df_content['Category'].values
p = df_content['Facebook Connected'].values
q = df_content['Has Video'].values
r = df_content['Creator Website'].values

X = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r])
X = X.transpose()

y = np.array([df_content['Backers'].values])
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
# Saving model to disk
#pickle.dump(reg, open('back-content.pkl','wb'))
b_content=pickle.load(open('back-content.pkl', 'rb'))
a_content=pickle.load(open('ap-content.pkl', 'rb'))
print(b_content.predict([[6500,7,60,0,0]]))

print(a_content.predict([[6500,7,60,0,0]]))