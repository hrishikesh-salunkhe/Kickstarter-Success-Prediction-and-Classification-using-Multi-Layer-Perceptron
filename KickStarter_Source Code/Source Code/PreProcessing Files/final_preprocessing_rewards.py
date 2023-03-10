# -*- coding: utf-8 -*-
"""FINAL_preprocessing_rewards.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d7iz35UoZI-teC42pdsB3ssG6aQ5sdXh
"""

from google.colab import drive
drive.mount('/content/gdrive')

!cp "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards.xlsx" "/content/kickstarter-rewards.xlsx"

!mkdir "/content/gdrive/My Drive/Datasets/"
!mkdir "/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/"

import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt 
import datetime
import xlrd

df = pd.read_excel("kickstarter-rewards.xlsx")

df = df[['project id', 'category', 'subcategory',
       'status', 'goal', 'pledged', 'backers',
       'levels', 'reward levels', 'updates', 'comments',
       'duration']]
df

df = df.set_index("status")
df = df.drop(['canceled', 'live', 'suspended'], axis=0)

df = df.reset_index()

"""# **Unique values function**"""

def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
        
    return unique_list

unique(df["subcategory"])

len(df)

"""# ***Load the arrays if preprocessing has already been done***"""

status = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/status.npy')
subcategory = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/subcategory.npy')
category = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/category.npy')
min_reward = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/min_reward.npy')
max_reward = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/max_reward.npy')
mean_reward = np.load('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/mean_reward.npy')

"""# **Preprocessing Starts!**"""

status = []

for i in range(0,len(df)):
  if(df.status[i]=="failed"):
    status.append(0)
  else:
    status.append(1)
    
print(status)

category =[]

unique_list = unique(df["category"])

for i in range(0,len(df)):
  for j in range(0, len(unique_list)):
    if(df["category"][i]==unique_list[j]):
      category.append(j)
     
print(category)

subcategory =[]

unique_list = unique(df["subcategory"])

for i in range(0,len(df)):
  for j in range(0, len(unique_list)):
    if(df["subcategory"][i]==unique_list[j]):
      subcategory.append(j)
      
print(subcategory)

reward_levels = []

for i in range(0, len(df)):
  
  if(type(df["reward levels"][i]) == str):
    temp = df["reward levels"][i].split("$")
    temp = temp[1:]
    for j in range(0, len(temp)):
      temp[j] = int(temp[j].replace(',', ''))
    reward_levels.append(temp)
  else:
    reward_levels.append([])
    
min_reward = []
max_reward = []
mean_reward = []

for i in range(0, len(reward_levels)):
  if not reward_levels[i]:
    min_reward.append(0)
    max_reward.append(0)
    mean_reward.append(0)
  else:
    min_reward.append(reward_levels[i][0])
    max_reward.append(reward_levels[i][-1])
    mean_reward.append(sum(reward_levels[i])/float(len(reward_levels[i])))

print(reward_levels)
print(min_reward)
print(max_reward)
print(mean_reward)

np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/status.npy', status)
np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/category.npy', category)
np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/subcategory.npy', subcategory)
np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/min_reward.npy', min_reward)
np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/max_reward.npy', max_reward)
np.save('/content/gdrive/My Drive/Datasets/kickstarter-rewards-info/mean_reward.npy', mean_reward)

"""# **Merging the arrays into the main dataframe**"""

df_temp = df[['project id', 'goal', 'pledged', 'backers', 'levels', 'updates', 'comments', 'duration']]

data = {'project id': df_temp["project id"], 'status': status, 'category': category,
        'subcategory': subcategory, 'min reward': min_reward, 'max reward': max_reward, 'mean reward': mean_reward}

df_array = pd.DataFrame(data)
df_final = pd.merge(df_temp,df_array,on="project id")

"""# **Removing null values from the final dataset**"""

df_final = df_final.fillna(99999)

df_final.corr()

"""# OUTLIER DETECTION:

Enter the number of rounds of Outlier Removal to be performed below:
"""

n_rounds = 10

def find_anomalies(random_data):
    anomalies =[]
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    #print(lower_limit)
    # Generate outliers
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

for i in range(n_rounds):
  # 1: GOAL

  temp = unique(find_anomalies(df_final['goal']))

  df_final = df_final.set_index("goal")
  df_final = df_final.drop(temp, axis = 0)

  df_final = df_final.reset_index()

  # # 2: LEVELS

  # temp = unique(find_anomalies(df_final['levels']))

  # df_final = df_final.set_index("levels")
  # df_final = df_final.drop(temp, axis = 0)

  # df_final = df_final.reset_index()

  # # 3: DURATION

  # temp = unique(find_anomalies(df_final['duration']))

  # df_final = df_final.set_index("duration")
  # df_final = df_final.drop(temp, axis = 0)

  # df_final = df_final.reset_index()

  # # 4: CATEGORY

  # temp = unique(find_anomalies(df_final['category']))

  # df_final = df_final.set_index("category")
  # df_final = df_final.drop(temp, axis = 0)

  # df_final = df_final.reset_index()

  # # 4: SUBCATEGORY

  # temp = unique(find_anomalies(df_final['subcategory']))

  # df_final = df_final.set_index("subcategory")
  # df_final = df_final.drop(temp, axis = 0)

  # df_final = df_final.reset_index()

  # 5: MIN REWARD

  temp = unique(find_anomalies(df_final['min reward']))

  df_final = df_final.set_index("min reward")
  df_final = df_final.drop(temp, axis = 0)

  df_final = df_final.reset_index()

  # 6: MAX REWARD

  temp = unique(find_anomalies(df_final['max reward']))

  df_final = df_final.set_index("max reward")
  df_final = df_final.drop(temp, axis = 0)

  df_final = df_final.reset_index()

  # 7: MEAN REWARD

  temp = unique(find_anomalies(df_final['mean reward']))

  df_final = df_final.set_index("mean reward")
  df_final = df_final.drop(temp, axis = 0)

  df_final = df_final.reset_index()

  print('ROUND ', i+1)
  print('Number of rows left in the dataset: ', len(df_final))

df_final.corr()

#         project id	goal	   pledged	backers	   levels	  updates	  comments	 duration	 status	   category	subcategory	min reward	max reward	mean reward
# status	-0.000222	-0.036622	0.065098	0.074622	0.142421	0.389515	0.038072	-0.140601	1.000000	-0.074196	-0.010917	  -0.025010	  -0.075054	  -0.104544

#         mean reward	max reward	min reward	subcategory	 category	 duration	levels	     goal	  project id	pledged	  backers	updates	comments	status
# status	0.060763    	0.066595	 0.027606	   -0.026176	-0.039970	-0.136362	0.159430	-0.201622	-0.000060 	0.308425	0.244435	0.373827	0.149095	1.000000

#         subcategory	 category	  duration	     levels	    goal	  project id	pledged	 backers	 updates	 comments	 status	  min reward	max reward	mean reward
# status	-0.017278	   -0.047787	-0.130309	    0.159073	-0.195520	-0.003244	 0.268677	 0.223693	 0.383094	  0.100336	1.000000	-0.036370	-0.015193	-0.033590

#        mean reward	max reward	min reward	subcategory	category	duration	levels	    goal	project id	pledged	backers	updates	comments	status
# status	0.067548	   0.077804	   0.025348	  -0.024934	 -0.037605	-0.120585	0.158996	-0.200416	0.000905	0.287288	0.230831	0.371434	0.141460	1.000000

#        subcategory	category	levels	     goal	  project id	pledged	  backers	  updates	  comments	 duration	status	  min reward	max reward	mean reward
# status	-0.017278	 -0.047787	0.159073	-0.195520	-0.003244	  0.268677	0.223693	0.383094	0.100336	-0.130309	1.000000	-0.036370	  -0.015193	  -0.033590

#       subcategory	 category	levels	  project id	goal	  pledged	  backers	  updates	  comments	 duration	 status	  min reward	max reward	mean reward
# status -0.013415	-0.071100	0.140453	-0.000919	-0.036370	0.060817	0.070734	0.391548	0.040832	-0.142171	1.000000	-0.024906	  -0.083310	  -0.107125

"""# **Implementing Classifier**"""

a = df_final['goal'].values
b = df_final['levels'].values
c = df_final['duration'].values
d = df_final['category'].values
e = df_final['subcategory'].values
f = df_final['min reward'].values
g = df_final['max reward'].values
h = df_final['mean reward'].values

X = np.array([a,b,c,d,e,f,g,h])
X = X.transpose()
print(X.shape)

np.savetxt("kickstarter_rewards_inputs.csv", X, delimiter=",")
!cp "/content/kickstarter_rewards_inputs.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_inputs.csv"

s = df_final['status'].values
y = np.array([s])
y = y.transpose()
y = y.ravel()
print(y.shape)

np.savetxt("kickstarter_rewards_labels.csv", y, delimiter=",")
!cp "/content/kickstarter_rewards_labels.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_labels.csv"

"""# ***Generating (and Saving) a csv file of all the final values to be used***"""

data_final = {'goal': a, 'levels': b, 'duration': c, 'category': d,
        'subcategory': e, 'min reward': f, 'max reward': g, 'mean reward': h, 'status': s}

data_final_lr = {'goal': a, 'pledged': df_final['pledged'].values, 'backers': df_final['backers'].values, 'levels': b, 'duration': c, 'category': d,
        'subcategory': e, 'min reward': f, 'max reward': g, 'mean reward': h, 'status': s}

df_array_final = pd.DataFrame(data_final)
df_array_final_sample = df_array_final.head(1000)

df_array_final_lr = pd.DataFrame(data_final_lr)
df_array_final_lr_sample = df_array_final_lr.head(1000)

df_array_final_lr.to_csv (r'kickstarter_rewards_final_lr_header.csv', index = False, header=True)
df_array_final_lr_sample.to_csv (r'kickstarter_rewards_final_lr_sample_header.csv', index = False, header=True)

df_array_final.to_csv (r'kickstarter_rewards_final.csv', index = False, header=False)
df_array_final_sample.to_csv (r'kickstarter_rewards_final_sample.csv', index = False, header=False)

df_array_final.to_csv (r'kickstarter_rewards_final_header.csv', index = False, header=True)
df_array_final_sample.to_csv (r'kickstarter_rewards_final_sample_header.csv', index = False, header=True)

!cp "/content/kickstarter_rewards_final.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final.csv"
!cp "/content/kickstarter_rewards_final_sample.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final_sample.csv"

!cp "/content/kickstarter_rewards_final_header.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final_header.csv"
!cp "/content/kickstarter_rewards_final_sample_header.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final_sample_header.csv"

!cp "/content/kickstarter_rewards_final_lr_header.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final_lr_header.csv"
!cp "/content/kickstarter_rewards_final_lr_sample_header.csv" "/content/gdrive/My Drive/BE Project/Datasets/kickstarter-rewards-info/kickstarter_rewards_final_lr_sample_header.csv"