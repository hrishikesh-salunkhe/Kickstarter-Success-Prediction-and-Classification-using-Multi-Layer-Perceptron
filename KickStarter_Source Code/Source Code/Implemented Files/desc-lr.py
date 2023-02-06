# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:16:24 2020

@author: VyomeshS
"""


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

df_description = pd.read_csv('kickstarter_description.csv')
df_description.columns = ['n_words', 'n_sents', 'n_chars', 'n_syllables', 'n_unique_words',
        'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words',
        'flesch_kincaid_grade_level', 'flesch_reading_ease', 'smog_index',
        'gunning_fog_index', 'coleman_liau_index',
        'automated_readability_index', 'lix', 'wiener_sachtextformel', 'goal',
        'pledged', 'backers', 'category', 'main category', 'country',
        'currency', 'duration', 'state']

df_description = df_description.set_index("state")
df_description = df_description.drop([0], axis=0)
df_description = df_description.reset_index()

a = df_description['n_words'].values
b = df_description['n_sents'].values
c = df_description['n_chars'].values
d = df_description['n_syllables'].values
e = df_description['n_unique_words'].values
f = df_description['n_long_words'].values
g = df_description['n_monosyllable_words'].values
h = df_description['n_polysyllable_words'].values
i = df_description['flesch_kincaid_grade_level'].values
j = df_description['flesch_reading_ease'].values
k = df_description['smog_index'].values
l = df_description['gunning_fog_index'].values
m = df_description['coleman_liau_index'].values
n = df_description['automated_readability_index'].values
o = df_description['lix'].values
p = df_description['wiener_sachtextformel'].values
q = df_description['goal'].values
r = df_description['category'].values
s = df_description['main category'].values
t = df_description['country'].values
u = df_description['currency'].values
v = df_description['duration'].values

X = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v])
X = X.transpose()

y = np.array([df_description['pledged'].values])
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
pickle.dump(reg, open('ap-desc.pkl','wb'))

