#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_tree


# In[23]:


table = pd.read_csv('xgboost_data')


# In[24]:


table.shape


# In[27]:


table = table.rename(columns={"AUC Train": "AUC_Train", "AUC Test 1": "AUC_TEST_1", "AUC Test 2": "AUC_TEST_2"})


# In[28]:


table


# In[29]:


table.AUC_TEST_1[0]


# In[ ]:


pd.set_option('display.max_rows', None)
table


# In[32]:


table.describe().transpose()


# In[35]:


table[table['AUC_TEST_2']>0.94]


# ### BEST PARAMETERS
#    * n_estimators: 300
#    * learning_rate: 0.1
#    * subsample: 0.8
#    * colsample_bytree: 0.5
#    * scale_pos_weight: 1

# In[36]:


train1 = pd.read_csv('train1')
test1 = pd.read_csv('test1')
test2 = pd.read_csv('test2')


# In[37]:


train1.head(5)


# In[38]:


X_train = train1.drop(['target'], axis = 1)
Y_train = train1['target']
X_test1 = test1.drop(['target'], axis = 1)
Y_test1 = test1['target']
X_test2 = test2.drop(['target'], axis = 1)
Y_test2 = test2['target']


# In[39]:


feature1 = pd.read_csv('features_default')
feature2 = pd.read_csv('features_parameters')
final_feature = pd.concat([feature1,feature2])
final_feature = final_feature.Feature.unique()


# In[32]:


feature = pd.read_csv('final_features')


# In[33]:


feature


# In[11]:


final_feature


# In[40]:


X_train = X_train[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48','D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test1 = X_test1[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]
X_test2 = X_test2[['P_2', 'B_1', 'B_9', 'D_44', 'D_42', 'B_3', 'S_3', 'D_45', 'D_48', 'D_51', 'R_1', 'B_7', 'D_66', 'R_27', 'D_43', 'B_2', 'B_11','D_50', 'B_38', 'D_132', 'B_4', 'D_41', 'S_23', 'D_62', 'B_39','D_75', 'D_77', 'B_18', 'D_46', 'B_5', 'R_3', 'D_49', 'R_26','D_56', 'CO', 'S_2', 'B_6', 'B_10', 'D_61', 'D_52', 'S_7', 'O','D_55', 'D_112', 'P_3']]


# In[41]:


X_train.head()


# In[42]:


X_test1.head()


# In[43]:


X_test2.head()


# In[44]:


X_train.shape


# In[45]:


xgb_instance = xgb.XGBClassifier(n_estimators=300, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.5, scale_pos_weight = 1, random_state = 7)
final_model = xgb_instance.fit(X_train, Y_train)


# In[119]:


final_model.predict(X_train)


# In[120]:


final_model.predict_proba(X_train)


# ## SHAP

# In[114]:


import shap
shap.initjs()
explainer = shap.Explainer(final_model)


# In[115]:


shap_values = explainer(X_test2)


# In[116]:


shap_values


# In[117]:


shap.plots.beeswarm(shap_values)


# In[118]:


shap.plots.waterfall(shap_values[397])


# ## Bad rate for train

# In[87]:


perf_train_data = pd.DataFrame({"Actual": train1['target'], "Prediction": final_model.predict_proba(X_train)[:,1]})
quantiles = list(set(perf_train_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"]).reset_index()
stat["Bad Rate"] = stat["sum"] / stat["count"]
stat['Score Bins'] = stat['Score Bins'].astype(str)
stat


# In[110]:


plt.bar(stat['Score Bins'],stat['Bad Rate'])
plt.xticks(rotation=90)
xlab = 'Score Bins'
ylab = 'Bad Rate'
title = 'Rank Orderings (TRAIN)'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
for index, value in enumerate(stat['Bad Rate']):
    plt.text(index-0.5, value,
             str(round(value,3)))
plt.show()


# ## Bad rate for test1 & test2

# In[91]:


perf_train_data = pd.DataFrame({"Actual": test1['target'], "Prediction": final_model.predict_proba(X_test1)[:,1]})

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat1 = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"]).reset_index()
stat1["Bad Rate"] = stat1["sum"] / stat1["count"]
stat1['Score Bins'] = stat1['Score Bins'].astype(str)
stat1


# In[111]:


plt.bar(stat1['Score Bins'],stat1['Bad Rate'])
plt.xticks(rotation=90)
xlab = 'Score Bins'
ylab = 'Bad Rate'
title = 'Rank Orderings (TEST 1)'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
for index, value in enumerate(stat1['Bad Rate']):
    plt.text(index-0.5, value,
             str(round(value,3)))
plt.show()


# In[107]:


perf_train_data = pd.DataFrame({"Actual": test2['target'], "Prediction": final_model.predict_proba(X_test2)[:,1]})

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat2 = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"]).reset_index()
stat2["Bad Rate"] = stat2["sum"] / stat2["count"]
stat2['Score Bins'] = stat2['Score Bins'].astype(str)
stat2


# In[112]:


plt.bar(stat2['Score Bins'],stat2['Bad Rate'])
plt.xticks(rotation=90)
xlab = 'Score Bins'
ylab = 'Bad Rate'
title = 'Rank Orderings (TEST 2)'
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
for index, value in enumerate(stat2['Bad Rate']):
    plt.text(index-0.5, value,
             str(round(value,3)))
plt.show()


# In[93]:


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(Y_train,final_model.predict(X_train)))
accuracy_score(Y_train,final_model.predict(X_train))


# In[94]:


print(confusion_matrix(Y_test1,final_model.predict(X_test1)))
accuracy_score(Y_test1,final_model.predict(X_test1))


# In[95]:


print(confusion_matrix(Y_test2,final_model.predict(X_test2)))
accuracy_score(Y_test2,final_model.predict(X_test2))


# ## SAVE FINAL MODEL

# In[102]:


final_model.save_model("final_model.json")

