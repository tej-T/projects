#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


# In[2]:


train1 = pd.read_csv('train1')
test1 = pd.read_csv('test1')
test2 = pd.read_csv('test2')


# In[3]:


X_train = train1.drop(['target'], axis = 1)
Y_train = train1['target']
X_test1 = test1.drop(['target'], axis = 1)
Y_test1 = test1['target']
X_test2 = test2.drop(['target'], axis = 1)
Y_test2 = test2['target']


# In[4]:


X_train = X_train[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test1 = X_test1[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test2 = X_test2[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]


# In[12]:


xgb_instance = xgb.XGBClassifier(n_estimators=300, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.5, scale_pos_weight = 1, random_state = 7)
final_model = xgb_instance.fit(X_train, Y_train)


# In[13]:


X_train['y_pred'] = final_model.predict_proba(X_train)[:,1]
X_test1['y_pred'] = final_model.predict_proba(X_test1)[:,1]
X_test2['y_pred'] = final_model.predict_proba(X_test2)[:,1]


# In[14]:


X_train = X_train[['B_1', 'S_23',"y_pred"]]
X_test1 = X_test1[['B_1', 'S_23',"y_pred"]]
X_test2 = X_test2[['B_1', 'S_23',"y_pred"]]


# In[15]:


X_train["Default"] = 0
X_test1["Default"] = 0
X_test2["Default"] = 0
X_train["Rev"] = 0
X_test1["Rev"] = 0
X_test2["Rev"] = 0


# In[16]:


X_train


# In[17]:


X_test1


# In[18]:


X_test2


# In[19]:


X_train.to_csv("xtrain")


# In[33]:


X_test1.to_csv("xt1")
X_test2.to_csv("xt2")


# ## BUILD FUNCTION FOR DEFAULT RATE AND REVENUE
# #### INPUTS :
# * Sample
# * Target Var ('TARGET')
# * THRESHOLD

# In[39]:


def rev_gen(X,Y,thres):
    model = xgb.XGBClassifier(n_estimators=300, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.5, scale_pos_weight = 1, random_state = 7)
    model.load_model("final_model.json")
    X_sam = X.drop([Y], axis = 1)
    X_sam = X_sam[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
    X_sam['y_pred'] = model.predict_proba(X_sam)[:,1]
    X_sam = X_sam[['B_1', 'S_23',"y_pred"]]
    Y_sam = X[Y]
    sam = pd.concat([X_sam,Y_sam],axis = 1)
    sam = sam[sam['y_pred']<thres]
    total = len(sam)
    sam2 = sam[sam['target']==1]
    default = len(sam2)
    default_rate = default/total
    print("The Total is ",total)
    print("The Default rate is ",default_rate)
    sam3 = sam[sam['target']==0]
    spend = sam3['S_23'].sum()
    balance = sam3['B_1'].sum()
    revenue = (balance*0.02)+(spend*0.01)
    print("The revenue is ",revenue)


# In[21]:


rev_gen(train1,'target',0.1)


# In[22]:


rev_gen(train1,'target',0.81)


# In[23]:


rev_gen(train1,'target',0.44)


# In[41]:


rev_gen(train1,'target',.58)


# In[42]:


rev_gen(test1,'target',.58)


# In[43]:


rev_gen(test2,'target',.58)


# In[44]:


rev_gen(Data,'target',.58)


# In[45]:


rev_gen(train1,'target',.35)


# In[46]:


rev_gen(test1,'target',.35)


# In[47]:


rev_gen(test2,'target',.35)


# In[48]:


rev_gen(Data,'target',.35)


# ## GENERATE TABLE

# In[24]:


def table_gen(X_sam,Y_sam,thres):
    sam = pd.concat([X_sam,Y_sam],axis = 1)
    sam = sam[sam['y_pred']<thres]
    total = len(sam)
    sam2 = sam[sam['target']==1]
    default = len(sam2)
    default_rate = default/total
    sam3 = sam[sam['target']==0]
    spend = sam3['S_23'].sum()
    balance = sam3['B_1'].sum()
    revenue = (balance*0.02)+(spend*0.01)
    Sample_1.loc[row,"Threshold"] = thres
    Sample_1.loc[row,"Total"] = total
    Sample_1.loc[row,"Default"] = default
    Sample_1.loc[row,"Default_Rate"] = default_rate
    Sample_1.loc[row,"Revenue"] = revenue


# In[25]:


# WRITE TRAIN DATA
Sample_1 = pd.DataFrame(columns = ["Threshold", "Total", "Default", "Default_Rate", "Revenue"])
row = 0
length = 10
for t in range(1, length+1):
    table_gen(X_train,Y_train,t/length)
    row = row + 1
Sample_1


# In[26]:


Sample_1.to_csv('SAMPLE_1.csv')


# In[27]:


# WRITE TEST1 DATA
Sample_1 = pd.DataFrame(columns = ["Threshold", "Total", "Default", "Default_Rate", "Revenue"])
row = 0
length = 10
for t in range(1, length+1):
    table_gen(X_test1,Y_test1,t/length)
    row = row + 1
Sample_1


# In[28]:


Sample_1.to_csv('SAMPLE_TEST1.csv')


# In[29]:


# WRITE TEST2 DATA
Sample_1 = pd.DataFrame(columns = ["Threshold", "Total", "Default", "Default_Rate", "Revenue"])
row = 0
length = 10
for t in range(1, length+1):
    table_gen(X_test2,Y_test2,t/10)
    row = row + 1
Sample_1


# In[30]:


Sample_1.to_csv('SAMPLE_TEST2.csv')


# In[31]:


Data = pd.read_csv('Total_data')


# In[32]:


X = Data.drop(['target'], axis = 1)
Y = Data['target']
X = X[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X['y_pred'] = final_model.predict_proba(X)[:,1]
X = X[['B_1', 'S_23',"y_pred"]]


# In[33]:


Sample_1 = pd.DataFrame(columns = ["Threshold", "Total", "Default", "Default_Rate", "Revenue"])
row = 0
length = 10
for t in range(1, length+1):
    table_gen(X,Y,t/length)
    row = row + 1
Sample_1


# In[34]:


Sample_1.to_csv('SAMPLE_TOTAL.csv')


# In[35]:


Sample_1 = pd.DataFrame(columns = ["Threshold", "Total", "Default", "Default_Rate", "Revenue"])
row = 0
length = 100
for t in range(1, length+1):
    table_gen(X,Y,t/length)
    row = row + 1
Sample_1


# In[36]:


Sample_1.to_csv('SAMPLE_TOTAL_STRATEGY.csv')


# In[37]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    display(Sample_1)


# ## AGRESSIVE STRATEGY : 58% THRESHOLD
# ## CONSERVATIVE STRATEGY : 35% THRESHOLD
