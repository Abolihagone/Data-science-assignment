#!/usr/bin/env python
# coding: utf-8

# # Neural Network(Forest Fire)

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#Import dataset
forest_fire=pd.read_csv("C:/Users/Aboli/Downloads/forestfires.csv")
forest_fire


# In[6]:


forest_fire.shape


# In[8]:


forest_fire.info()


# In[9]:


forest_fire.isnull().sum()


# In[11]:


sns.countplot(x='size_category',data =forest_fire)


# In[13]:


plt.figure(figsize=(20,10))
sns.barplot(x='month',y='temp',data=forest_fire,
            order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
plt.title('Month Vs Temp')
plt.xlabel('month')
plt.ylabel('temp')


# In[15]:


#Dropping columns which are not required
forest_fire=forest_fire.drop(columns=['dayfri','daymon','daysat','daysun','daythu','daytue','daywed','monthapr',	
                               'monthaug','monthdec','monthfeb','monthjan','monthjul','monthjun','monthmar',
                               'monthmay','monthnov','monthoct','monthsep'],axis=1)


# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(forest_fire.corr(),annot=True,cmap="inferno")
plt.title("HeatMap of Features for the Classes")


# In[20]:


forest_fire["month"].value_counts()


# In[22]:


month_data={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}
forest_fire.replace(month_data,inplace=True)


# In[23]:


forest_fire['size_category'].unique()


# In[25]:


forest_fire.size_category.replace(('small','large'),(1,0),inplace=True)


# In[27]:


forest_fire["day"].value_counts()


# In[29]:


day_data={'day':{'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}}
forest_fire.replace(day_data,inplace=True)


# In[31]:


X=forest_fire.iloc[:,0:11]
X


# In[32]:


Y=forest_fire["size_category"]
Y


# In[33]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
x_scaled


# In[34]:


scaled_ff=pd.DataFrame(x_scaled,columns=X.columns)
scaled_ff


# In[35]:


X_transformed=scaled_ff
X_transformed


# In[36]:


Y


# In[38]:


X_train,X_test,Y_train,Y_test=train_test_split(X_transformed,Y,test_size=0.20,random_state=123)


# In[39]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[40]:


model=Sequential()
model.add(Dense(units=12,input_dim=11,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=10,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))


# In[41]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[42]:


model.fit(X_train,Y_train, epochs=100, batch_size=10)


# In[43]:


scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))


# In[ ]:





# # Neural Networking (Gas Turbine)

# In[44]:


#Import Libraries
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt


# In[45]:


#Import dataset
gas = pd.read_csv("C:/Users/Aboli/Downloads/gas_turbines (1).csv")
gas


# In[46]:


gas.shape


# In[47]:


gas.info()


# In[48]:


gas.isnull().sum()


# In[49]:


gas.corr()


# In[50]:


plt.figure(figsize=(10,10))
sns.heatmap(gas.corr(),annot=True,cmap="inferno")


# In[51]:


X=gas.drop('TEY',axis=1)
X


# In[52]:


Y=gas['TEY']
Y


# In[53]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
x_scaled


# In[54]:


scaled_gasturbines_data=pd.DataFrame(x_scaled,columns=X.columns)
scaled_gasturbines_data


# In[55]:


X_transformed=scaled_gasturbines_data
X_transformed


# In[56]:


X_train,X_test,Y_train,Y_test=train_test_split(X_transformed,Y,test_size=0.20,random_state=123)


# In[57]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[58]:


model=Sequential()
model.add(Dense(units=10,input_dim=10,activation ='relu',kernel_initializer='normal'))
model.add(Dense(units=6,activation='tanh',kernel_initializer='normal'))
model.add(Dense(units=1,activation='relu',kernel_initializer='normal'))


# In[59]:


model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mse'])


# In[60]:


model.fit(X_train,Y_train, epochs=100, batch_size=20)


# In[61]:


scores = model.evaluate(X_test,Y_test)
print((model.metrics_names[1]))


# # The End

# In[ ]:




