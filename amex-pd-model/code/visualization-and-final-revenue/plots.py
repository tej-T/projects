#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('scatter_plottable')


# In[3]:


data = data.rename(columns={"AUC Train": "AUC_Train", "AUC Test 1": "AUC_TEST_1", "AUC Test 2": "AUC_TEST_2"})
AUC_AVG = (data['AUC_Train'] + data['AUC_TEST_1'] + data['AUC_TEST_2'])/3
AUC_SD = []
for i in range(72):
    AUC_SD.append(np.std([data.AUC_Train[i],data.AUC_TEST_1[i],data.AUC_TEST_2[i]]))
data['AUC_AVG'] = AUC_AVG
data['AUC_SD'] = AUC_SD


# In[4]:


data


# In[5]:


plt.figure(figsize=(12,7))
plt.scatter(data.AUC_AVG, data.AUC_SD)
xlab = 'Average of AUC'
ylab = 'Standard Deviation of AUC'
title = 'AVG vs SD'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.show()


# In[6]:


plt.figure(figsize=(12,7))
plt.scatter(data.AUC_Train, data.AUC_TEST_2)
xlab = 'AUC of Train sample'
ylab = 'AUC of Test 2'
title = 'Train vs Test 2 AUC'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.axvline(x=0.948665)
plt.axhline(y=0.941923)
plt.show()


# In[7]:


data = pd.read_csv('NN_data')


# In[8]:


data = data.rename(columns={"AUC Train": "AUC_Train", "AUC Test 1": "AUC_TEST_1", "AUC Test 2": "AUC_TEST_2"})
AUC_AVG = (data['AUC_Train'] + data['AUC_TEST_1'] + data['AUC_TEST_2'])/3
AUC_SD = []
for i in range(32):
    AUC_SD.append(np.std([data.AUC_Train[i],data.AUC_TEST_1[i],data.AUC_TEST_2[i]]))
data['AUC_AVG'] = AUC_AVG
data['AUC_SD'] = AUC_SD


# In[9]:


data


# In[11]:


plt.figure(figsize=(12,7))
plt.scatter(data.AUC_AVG, data.AUC_SD)
xlab = 'Average of AUC'
ylab = 'Standard Deviation of AUC'
title = 'AVG vs SD (NEURAL NETWORK)'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.show()


# In[12]:


plt.figure(figsize=(12,7))
plt.scatter(data.AUC_Train, data.AUC_TEST_2)
xlab = 'AUC of Train sample'
ylab = 'AUC of Test 2'
title = 'Train vs Test 2 AUC (NEURAL NETWORK)'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.axvline(x=0.929267)
plt.axhline(y=0.938644)
plt.show()


# In[20]:


df1 = pd.read_csv('feature1')
df2 = pd.read_csv('feature2')


# In[21]:


df1.to_csv('feature1.csv')


# In[22]:


df2.to_csv('feature2.csv')


# In[14]:


df1


# In[18]:


plt.figure(figsize=(12,7))
plt.bar(df1.Feature, df1.Importance)
xlab = 'Feature'
plt.xticks(rotation=80)
ylab = 'Importance'
title = 'Feature importance of default parameters'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.show()


# In[19]:


plt.figure(figsize=(12,7))
plt.bar(df2.Feature, df2.Importance)
xlab = 'Feature'
plt.xticks(rotation=80)
ylab = 'Importance'
title = 'Feature importance of parameters'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.show()


# In[32]:


final_feature = pd.concat([df1,df2])


# In[33]:


final_feature = final_feature['Feature'].unique()


# In[36]:


final_feature

