
# coding: utf-8

# In[153]:


import pandas as pd


# In[154]:


# read data
data = pd.read_csv('data_1.csv')
data.head()


# In[155]:


data.shape


# In[156]:


data['status'].value_counts()


# In[157]:


data.dtypes.value_counts()


# In[158]:


data.select_dtypes('object').apply(pd.Series.nunique, axis=0)


# In[159]:


data = data.drop(['trade_no', 'bank_card_no', 'source', 'id_name', 'latest_query_time'], axis=1)
data.head()


# In[160]:


# 缺失值处理
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " rows that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[161]:


missing_values = missing_values_table(data)
missing_values.head(20)


# In[162]:


data['student_feature'].describe()


# In[163]:


data['student_feature'] = data['student_feature'].fillna(1)


# In[164]:


cols = missing_values.index
list=['loans_latest_time','reg_preference_for_trad']
for col in cols:   
    if col not in list:
        data[col] = data[col].fillna(data[col].mean())


# In[165]:


data['loans_latest_time'] = data['loans_latest_time'].fillna(0)


# In[166]:


data['reg_preference_for_trad'] = data['reg_preference_for_trad'].fillna(0)


# In[167]:


missing_values = missing_values_table(data)
missing_values.head()


# In[168]:


data_label = data['status']
data = data.drop(['status'], axis=1)


# In[169]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, data_label, test_size=.3, random_state=2018)
print("Training Size:{}".format(x_train.shape))
print('Testing Size:{}'.format(x_test.shape))

