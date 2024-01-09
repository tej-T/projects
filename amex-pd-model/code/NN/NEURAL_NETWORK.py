#!/usr/bin/env python
# coding: utf-8

# ## IMPORT Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## IMPORT DATABASES

# In[2]:


train1 = pd.read_csv('train1')
test1 = pd.read_csv('test1')
test2 = pd.read_csv('test2')
X_train = train1.drop(['target'], axis = 1)
Y_train = train1['target']
X_test1 = test1.drop(['target'], axis = 1)
Y_test1 = test1['target']
X_test2 = test2.drop(['target'], axis = 1)
Y_test2 = test2['target']


# ## FROM THE FEATURES SELECTED IN XGBOOST

# In[3]:


X_train = X_train[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test1 = X_test1[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test2 = X_test2[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]


# ## NORMALIZE DATA

# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)


# In[5]:


X_train_normalized = sc.transform(X_train)
X_test1_normalized = sc.transform(X_test1)
X_test2_normalized = sc.transform(X_test2)


# In[6]:


# convert to Pandas DF
X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
X_test1_normalized = pd.DataFrame(X_test1_normalized, columns=X_test1.columns)
X_test2_normalized = pd.DataFrame(X_test2_normalized, columns=X_test2.columns)


# In[7]:


X_train_normalized.describe(percentiles=[0.01, 0.99]).transpose()


# ## OUTLIER TREATMENT

# In[8]:


arr = ['B_9','S_3','D_48','D_43','D_50','D_132','S_23','D_62','D_77','D_46','B_5','R_3','D_49','R_26','D_56','B_6','B_10','D_61','D_41']


# In[9]:


for i in arr:
    X_train_normalized[i] = np.where((X_train_normalized[i] > X_train_normalized[i].quantile(0.99)), X_train_normalized[i].quantile(0.99), X_train_normalized[i])  
X_train_normalized['S_23'] = np.where((X_train_normalized['S_23'] < X_train_normalized['S_23'].quantile(0.01)), X_train_normalized['S_23'].quantile(0.01), X_train_normalized['S_23'])
X_train_normalized['D_46'] = np.where((X_train_normalized['D_46'] < X_train_normalized['D_46'].quantile(0.01)), X_train_normalized['D_46'].quantile(0.01), X_train_normalized['D_46'])


# In[10]:


X_train_normalized.describe(percentiles=[0.01, 0.99]).transpose()


# In[11]:


for i in arr:
    X_test1_normalized[i] = np.where((X_test1_normalized[i] > X_test1_normalized[i].quantile(0.99)), X_test1_normalized[i].quantile(0.99), X_test1_normalized[i])  
X_test1_normalized['S_23'] = np.where((X_test1_normalized['S_23'] < X_test1_normalized['S_23'].quantile(0.01)), X_test1_normalized['S_23'].quantile(0.01), X_test1_normalized['S_23'])
X_test1_normalized['D_46'] = np.where((X_test1_normalized['D_46'] < X_test1_normalized['D_46'].quantile(0.01)), X_test1_normalized['D_46'].quantile(0.01), X_test1_normalized['D_46'])
X_test1_normalized.describe(percentiles=[0.01, 0.99]).transpose()


# In[12]:


for i in arr:
    X_test2_normalized[i] = np.where((X_test2_normalized[i] > X_test2_normalized[i].quantile(0.99)), X_test2_normalized[i].quantile(0.99), X_test2_normalized[i])  
X_test2_normalized['S_23'] = np.where((X_test2_normalized['S_23'] < X_test2_normalized['S_23'].quantile(0.01)), X_test2_normalized['S_23'].quantile(0.01), X_test2_normalized['S_23'])
X_test2_normalized['D_46'] = np.where((X_test2_normalized['D_46'] < X_test2_normalized['D_46'].quantile(0.01)), X_test2_normalized['D_46'].quantile(0.01), X_test2_normalized['D_46'])
X_test2_normalized.describe(percentiles=[0.01, 0.99]).transpose()


# ## REPLACE MISSING VALUES

# In[13]:


X_train_normalized.isnull().sum().sum()


# In[14]:


X_test1_normalized.isnull().sum().sum()


# In[15]:


X_test2_normalized.isnull().sum().sum()


# In[16]:


X_train_normalized = X_train_normalized.fillna(0)
X_test1_normalized = X_test1_normalized.fillna(0)
X_test2_normalized = X_test2_normalized.fillna(0)


# In[17]:


X_train_normalized.isnull().sum().sum()


# In[18]:


X_test1_normalized.isnull().sum().sum()


# In[19]:


X_test2_normalized.isnull().sum().sum()


# ## TRAIN MODEL

# In[20]:


import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


# ## GRID SEARCH

# In[29]:


from sklearn.metrics import roc_auc_score
table = pd.DataFrame(columns = ["Num Layers", "Nodes in Layer", "Activation Func", "Dropout Regularization", "Batch Size", "AUC Train", "AUC Test 1", "AUC Test 2"])

row = 0

for NL in [4,6]:
    for actfn in ['relu','tanh']:
        for dr in [0,0.5]:
            for btch in [100,10000]:
                classifier = Sequential()
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))
                classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
                model = classifier.fit(X_train_normalized,Y_train,batch_size=btch,epochs=20)
                table.loc[row,"Num Layers"] = 2
                table.loc[row,"Nodes in Layer"] = NL
                table.loc[row,"Activation Func"] = actfn
                table.loc[row,"Dropout Regularization"] = dr
                table.loc[row,"Batch Size"] = btch
                table.loc[row,"AUC Train"] = roc_auc_score(train1['target'], classifier.predict(X_train_normalized))
                table.loc[row,"AUC Test 1"] = roc_auc_score(test1['target'], classifier.predict(X_test1_normalized))
                table.loc[row,"AUC Test 2"] = roc_auc_score(test2['target'], classifier.predict(X_test2_normalized))
                
                row = row + 1

for NL in [4,6]:
    for actfn in ['relu','tanh']:
        for dr in [0,0.5]:
            for btch in [100,10000]:
                classifier = Sequential()
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=NL,kernel_initializer='glorot_uniform',activation = actfn))
                classifier.add(Dropout(dr))
                classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))
                classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
                model = classifier.fit(X_train_normalized,Y_train,batch_size=btch,epochs=20)
                table.loc[row,"Num Layers"] = 4
                table.loc[row,"Nodes in Layer"] = NL
                table.loc[row,"Activation Func"] = actfn
                table.loc[row,"Dropout Regularization"] = dr
                table.loc[row,"Batch Size"] = btch
                table.loc[row,"AUC Train"] = roc_auc_score(train1['target'], classifier.predict(X_train_normalized))
                table.loc[row,"AUC Test 1"] = roc_auc_score(test1['target'], classifier.predict(X_test1_normalized))
                table.loc[row,"AUC Test 2"] = roc_auc_score(test2['target'], classifier.predict(X_test2_normalized))
                
                row = row + 1
table


# In[31]:


table.to_csv('NN_data',index= False )


# In[32]:


table


# In[33]:


table.describe().transpose()


# In[35]:


table[table['AUC Test 2']>0.935]


# ### BEST MODEL
# classifier = Sequential()
#                 classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',activation = 'tanh'))
#                 classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',activation = 'tanh'))
#                 classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))
#                 classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#                 model = classifier.fit(X_train_normalized,Y_train,batch_size=100,epochs=20)

# ## Hyper parameters
# * NO of Layers = 2
# * Nodes in Layers = 6
# * Activation func = tanh
# * Dropout regularization = 0 (No dropout)
# * Batch size = 100

# In[36]:


classifier = Sequential()
classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',activation = 'tanh'))
classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',activation = 'tanh')) 
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model = classifier.fit(X_train_normalized,Y_train,batch_size=100,epochs=20)


# In[50]:


classifier.evaluate(X_test1_normalized, Y_test1)[1]


# In[51]:


classifier.evaluate(X_test2_normalized, Y_test2)[1]


# In[55]:


classifier.save("nn_final.json")


# ## FROM THE ACCURACY WE CAN STATE THAT XGBOOST IS THE BETTER MODEL
