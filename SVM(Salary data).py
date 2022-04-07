#!/usr/bin/env python
# coding: utf-8

# # SVM(Salary Data)

# In[1]:


# SVM Classification
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


salarytrain=pd.read_csv("C:/Users/Aboli/Downloads/SalaryData_Train(1).csv")
salarytest=pd.read_csv("C:/Users/Aboli/Downloads/SalaryData_Test(1).csv")


# In[3]:


salarytest


# In[4]:


salarytrain


# In[6]:


salarytest.info()


# In[7]:


salarytest.shape


# In[9]:


salarytrain.info()


# In[11]:


salarytrain.shape


# In[12]:


salarytrain.columns
salarytest.columns
stringcol=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[13]:


from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
for i in stringcol:
    salarytrain[i]=label_encoder.fit_transform(salarytrain[i])
    salarytest[i]=label_encoder.fit_transform(salarytest[i])


# In[14]:


salarytrain.head()


# In[15]:


salarytest.head()


# In[16]:


#coverting Y column in train test both 
salarytrain['Salary'] = label_encoder.fit_transform(salarytrain['Salary'])


# In[17]:


salarytest['Salary'] = label_encoder.fit_transform(salarytest['Salary'])


# In[18]:


salarytrain.head()


# In[19]:


salarytest.head()


# In[20]:


salarytrainx=salarytrain.iloc[:,0:13]
salarytrainy=salarytrain.iloc[:,13]
salarytestx=salarytest.iloc[:,0:13]
salarytesty=salarytest.iloc[:,13]


# In[21]:


trainx.shape ,trainy.shape, testx.shape, testy.shape


# # kernel = rbf

# In[24]:


#by rbf (radial basis function)
model_rbf=SVC(kernel='rbf')


# In[25]:


model_rbf.fit(salarytrainx,salarytrainy)


# In[ ]:


train_pred_rbf=model_rbf.predict(salarytrainx)
test_pred_rbf=model_rbf.predict(salarytestx)


# In[ ]:


train_rbf_acc=np.mean(train_pred_rbf==salarytrainy)
test_rbf_acc=np.mean(test_pred_rbf==salarytesty)


# In[ ]:


train_rbf_acc


# In[ ]:


test_rbf_acc

