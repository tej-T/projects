#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


# ## IMPORT DEV SET

# In[2]:


dev_set = pd.read_csv("devsample")


# In[30]:


dev_set.shape


# In[4]:


dev_set.head(5)


# In[5]:


dev_set.pop('count')
dev_set.pop('frac')
dev_set.pop('customer_ID')


# In[6]:


dev_set.shape


# In[7]:


dev_set.head(5)


# ## CHECK FOR NULL VALUES

# In[8]:


dev_set['target'].isnull().sum()


# In[9]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(dev_set.dtypes)


# GIVEN categorical features are:
# * B_30
# * B_38
# * D_63
# * D_64
# * D_66
# * D_68
# * D_114
# * D_116
# * D_117
# * D_120
# * D_126

# In[10]:


dev_set[['S_2','B_30','B_38','D_63','D_64','D_66','D_68','D_114','D_116','D_117','D_120','D_126']].head(15)


# ## ONE HOT ENCODING

# * D_* = Delinquency variables
# * S_* = Spend variables
# * P_* = Payment variables
# * B_* = Balance variables
# * R_* = Risk variables

# The variables that need one hot encoding are D_63 and D_64

# In[11]:


dev_set[['S_2','D_63']].groupby(["D_63"]).count()


# In[12]:


dev_set[['S_2','D_64']].groupby(["D_64"]).count()


# In[13]:


D_63_dummies = pd.get_dummies(dev_set.D_63)


# In[14]:


D_63_dummies.head(10)


# In[15]:


D_64_dummies = pd.get_dummies(dev_set.D_64)


# In[16]:


D_64_dummies.head(10)


# In[17]:


##replace -1 column with A (change column name)
dev_set.rename(columns = {'-1':'A'}, inplace = True)


# In[18]:


dev_set=pd.concat([dev_set, D_63_dummies], axis=1)


# In[19]:


dev_set=pd.concat([dev_set, D_64_dummies], axis=1)


# In[20]:


dev_set


# NOW REMOVE D_63 AND D_64. PLACE TARGET ON THE RIGHMOST COLUMN

# In[21]:


dev_set.pop('D_63')


# In[22]:


dev_set.pop('D_64')


# In[23]:


target = dev_set.pop('target')


# In[24]:


dev_set=pd.concat([dev_set, target], axis=1)


# In[25]:


dev_set.head(15)


# In[26]:


dev_set.shape


# In[27]:


dev_set = dev_set.sort_values(by=['S_2'])


# In[28]:


dev_set.head(15)


# In[29]:


dev_set.to_csv('Total_data')


# In[31]:


test1 = dev_set[dev_set['S_2'] < 201705]


# In[32]:


test1.shape


# In[33]:


test1.head()


# In[34]:


test2 = dev_set[dev_set['S_2'] > 201801]


# In[35]:


test2.shape


# In[36]:


train1 = dev_set[(dev_set['S_2'] <= 201801) & (dev_set['S_2'] >= 201705)]


# In[37]:


train1.shape


# ## EXPORT THE FILES

# In[37]:


test1.to_csv('test1', index=False)


# In[38]:


test2.to_csv('test2', index=False)


# In[39]:


train1.to_csv('train1', index=False)


# ## XGBOOST

# In[38]:


import xgboost as xgb


# In[39]:


X_train = train1.drop(['target'], axis = 1)
Y_train = train1['target']
X_test1 = test1.drop(['target'], axis = 1)
Y_test1 = test1['target']
X_test2 = test2.drop(['target'], axis = 1)
Y_test2 = test2['target']


# ## FEATURE SELECTION

# In[40]:


xgb_instance = xgb.XGBClassifier(random_state=7)
model_for_feature_selection = xgb_instance.fit(X_train, Y_train)


# In[41]:


feature_importance = {'Feature':X_train.columns,'Importance':model_for_feature_selection.feature_importances_}
feature_importance = pd.DataFrame(feature_importance)
feature_importance.sort_values("Importance", inplace=True,ascending=False)
feature_importance


# In[42]:


xgb_instance2 = xgb.XGBClassifier(n_estimators = 300, learning_rate = 0.5, max_depth = 4, subsample = 0.5, colsample_bytree = 0.5, scale_pos_weight = 5, random_state = 17)
model_for_feature_selection2 = xgb_instance2.fit(X_train, Y_train)
feature_importance2 = {'Feature':X_train.columns,'Importance':model_for_feature_selection2.feature_importances_}
feature_importance2 = pd.DataFrame(feature_importance2)
feature_importance2.sort_values("Importance", inplace=True,ascending=False)
feature_importance2


# In[43]:


feature_importance[feature_importance['Importance']>0.005]


# In[44]:


feature_importance2[feature_importance2['Importance']>0.005]


# In[49]:


final_features_default = feature_importance[feature_importance.Importance > 0.005]


# In[50]:


final_features_parameters = feature_importance2[feature_importance2.Importance > 0.005]


# In[52]:


final_features_default.to_csv('feature1')


# In[53]:


final_features_parameters.to_csv('feature2')


# In[45]:


final_features1 = feature_importance["Feature"][feature_importance.Importance > 0.005]


# In[46]:


final_features2 = feature_importance2["Feature"][feature_importance2.Importance > 0.005]


# In[51]:


final_feature = pd.concat([final_features1,final_features2])


# In[52]:


final_feature = final_feature.unique()


# In[53]:


final_features1.shape


# In[54]:


final_features2.shape


# In[55]:


final_feature.shape


# In[56]:


X_train = X_train[final_feature]
X_test1 = X_test1[final_feature]
X_test2 = X_test2[final_feature]


# In[57]:


X_train.head(10)


# In[58]:


X_train.shape


# In[59]:


X_test1.head(10)


# In[60]:


X_test2.head(10)


# In[61]:


X_test1.shape


# In[62]:


X_test2.shape


# ## GRID SEARCH

# In[66]:


from sklearn.metrics import roc_auc_score
table = pd.DataFrame(columns = ["Num Trees", "Learning Rate", "Subsample", "% in each tree", "Default weight", "AUC Train", "AUC Test 1", "AUC Test 2"])

row = 0
for num_trees in [50, 100, 300]:
  for LR in [0.01, 0.1]:
    for SS in [0.5, 0.8]:
        for sam in [0.5,1]:
            for wgts in [1,5,10]:
                xgb_instance = xgb.XGBClassifier(n_estimators=num_trees, learning_rate = LR ,subsample = SS, colsample_bytree = sam, scale_pos_weight = wgts )
                model = xgb_instance.fit(X_train, Y_train)
                table.loc[row,"Num Trees"] = num_trees
                table.loc[row,"Learning Rate"] = LR
                table.loc[row,"Subsample"] = SS
                table.loc[row,"% in each tree"] = sam
                table.loc[row,"Default weight"] = wgts
                table.loc[row,"AUC Train"] = roc_auc_score(train1['target'], model.predict_proba(X_train)[:,1])
                table.loc[row,"AUC Test 1"] = roc_auc_score(test1['target'], model.predict_proba(X_test1)[:,1])
                table.loc[row,"AUC Test 2"] = roc_auc_score(test2['target'], model.predict_proba(X_test2)[:,1])

                row = row + 1

table


# In[67]:


table.to_csv('xgboost_data',index =False)


# In[47]:


final_features1.to_csv('features_default',index = False)


# In[48]:


final_features2.to_csv('features_parameters',index = False)


# In[ ]:


final_feature = pd.DataFrame(final_feature)


# In[ ]:


final_feature.to_csv('final_features',index = False)


# In[72]:


table.head(35)


# In[73]:


table2 = table[table['AUC Train']>0.94]


# In[74]:


table2

