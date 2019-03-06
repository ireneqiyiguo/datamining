
# coding: utf-8

# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(data, data_label, test_size=.3, random_state=2018)


# In[68]:


# 逻辑回归
ltc = LogisticRegression()
ltc.fit(x_train, y_train)
predicted = ltc.predict(x_test)
accuracy_score(y_test, predicted)


# In[69]:


# svm
svc = SVC()
svc.fit(x_train, y_train)
predicted = svc.predict(x_test)
accuracy_score(y_test, predicted)


# In[70]:


# 决策树
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
predicted = dtc.predict(x_test)
accuracy_score(y_test, predicted)


# In[71]:


# 随机森林
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predicted = rfc.predict(x_test)
accuracy_score(y_test, predicted)


# In[72]:


# xgboost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
predicted = xgb.predict(x_test)
accuracy_score(predicted, y_test)

