#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit=pd.read_csv("creditcard.csv")
credit.head()


# In[3]:


credit.isnull().sum()


# In[4]:


credit.info()


# In[5]:


credit.shape


# In[6]:


credit['Class'].value_counts()


# # 0=normal transaction, 1= fradulant transaction
# 

# In[7]:


nor=credit[credit.Class==0]
fraud=credit[credit.Class==1]
print(nor.shape)
print(fraud.shape)


# In[8]:


nor.Amount.describe()


# In[9]:


fraud.Amount.describe()


# In[10]:


credit.groupby('Class').mean()


# # undersampling

# # build a  dataset similar to distribution of normal and fraud transaction

# In[11]:


nor_sample=nor.sample(n=492)


# In[12]:


new=pd.concat([nor_sample,fraud], axis=0)
new.head()


# In[13]:


new['Class'].value_counts()


# In[14]:


new.groupby('Class').mean()


# In[15]:


x=new.drop(columns='Class',axis=1)
y=new['Class']
print(x)


# In[16]:


print(y)


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)


# In[18]:


print(x.shape, x_train.shape ,x_test.shape)


# # LogisticRegression

# In[19]:


model=LogisticRegression()


# In[20]:


model.fit(x_train,y_train)


# In[21]:


predict=model.predict(x_train)
accuracy=accuracy_score(predict,y_train)
print("accuracy on training data", accuracy*100)


# In[22]:


predict1=model.predict(x_test)
accuracy1=accuracy_score(predict1,y_test)
print("accuracy on training data", accuracy1*100)

