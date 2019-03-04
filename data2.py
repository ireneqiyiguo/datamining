
# coding: utf-8


# In[112]:


num_reg={'一线城市':'1','二线城市':'2','三线城市':'3','其他城市':'4','境外':'5'}
data['reg_preference_for_trad']=data['reg_preference_for_trad'].map(num_reg)
data['reg_preference_for_trad']=data['reg_preference_for_trad'].fillna(0)
data['reg_preference_for_trad']=data['reg_preference_for_trad'].astype('float')


# In[138]:


import numpy as np


# In[146]:


data['loans_latest_time']=pd.to_datetime(data['loans_latest_time'],format='%Y/%m/%d').astype(np.int64)


# In[147]:


data['loans_latest_time'].head()


# In[148]:


data['avg_price_top_last_12_valid_month'] = data['avg_price_top_last_12_valid_month'].fillna(0)


# In[149]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, data_label, test_size=.3, random_state=2018)
print("Training Size:{}".format(x_train.shape))
print('Testing Size:{}'.format(x_test.shape))


# In[150]:


from sklearn.ensemble import RandomForestClassifier


# In[151]:


feat_lables = data.columns


# In[152]:


forest = RandomForestClassifier(n_estimators=10000, random_state=0,n_jobs=1)


# In[153]:


data_train = data.head(3327)


# In[155]:


rfc = RandomForestClassifier(random_state=2018)
rfc.fit(x_train,y_train)
importance = pd.Series(rfc.feature_importances_, index=x_train.columns).sort_values(ascending=False)
rfc_result = importance[: 20].index.tolist()
print(rfc_result)


# In[156]:


#计算IV的代码
def CalcIV(Xvar,Yvar):
    N_0=np.sum(Yvar==0)
    N_1=np.sum(Yvar==1)
    N_0_group=np.zeros(np.unique(Xvar).shape)
    
    N_1_group=np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        N_0_group[i] = Yvar[(Xvar==np.unique(Xvar)[i])&(Yvar==0)].count()
        N_1_group[i] = Yvar[(Xvar==np.unique(Xvar)[i])&(Yvar==1)].count()
    iv = np.sum((N_0_group/N_0-N_1_group/N_1)*np.log((N_0_group/N_0)/(N_1_group/N_1)))
    if iv>=1.0:## 处理极端值
       iv=1
    return iv

def caliv_batch(df,Yvar):
    ivlist=[]
    for col in df.columns:
        iv=CalcIV(df[col],Yvar)
        ivlist.append(iv)
    names=df.columns.values.tolist()
    iv_df=pd.DataFrame({'Var':names,'Iv':ivlist},columns=['Var','Iv'])
    return iv_df,ivlist
im_iv, ivl = caliv_batch(x_train,y_train)


# In[158]:


from xgboost import XGBClassifier
xgbc_model = XGBClassifier()
xgbc_model.fit(x_train, y_train)
xgbc_model_predict = xgbc_model.predict(x_test)


# In[ ]:


from sklearn.metrics import*
precision_score(y_test, xgbc_model_predict)

