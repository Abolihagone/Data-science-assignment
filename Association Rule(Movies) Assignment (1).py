#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


movies=pd.read_csv("C:/Users/Aboli/Downloads/my_movies.csv")


# In[3]:


movies.info()


# In[4]:


movies1=movies.drop(['V1','V2','V3','V4','V5'], axis=1)
movies1


# # Apriori Algorithm

# In[5]:


#with min_support of 0.1
frequent_itemsets1=apriori(movies1,min_support=0.1,use_colnames=True)
frequent_itemsets1


# In[6]:


rules1= association_rules(frequent_itemsets1, metric="lift", min_threshold=0.7)
rules1


# In[7]:


rules1.sort_values('lift',ascending = False)[0:20]


# In[8]:


rules1[rules1.lift>1]


# In[9]:


df1=pd.DataFrame(data=frequent_itemsets1)
df1
df1.duplicated()


# In[10]:


# visualization of obtained rule
rules1.plot(kind='bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[11]:


# visualization of obtained rule
plt.scatter(rules1.support,rules1.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[12]:


#with min_support of 0.2
frequent_itemsets2=apriori(movies1,min_support=0.2,use_colnames=True)
frequent_itemsets2


# In[13]:


rules2= association_rules(frequent_itemsets2, metric="lift", min_threshold=0.7)
rules2


# In[14]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[15]:


rules2[rules2.lift>1]


# In[16]:


df2=pd.DataFrame(data=frequent_itemsets2)
df2
df2.duplicated()


# In[17]:


# visualization of obtained rule
rules2.plot(kind='bar',x='support',y='confidence',color='black')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[18]:


# visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[19]:


#with min_support of 0.3
frequent_itemsets3=apriori(movies1,min_support=0.3,use_colnames=True)
frequent_itemsets3


# In[20]:


rules3= association_rules(frequent_itemsets3, metric="lift", min_threshold=0.7)
rules3


# In[21]:


rules3.sort_values('lift',ascending = False)[0:20]


# In[22]:


rules3[rules3.lift>1]


# In[23]:


df3=pd.DataFrame(data=frequent_itemsets3)
df3
df3.duplicated()


# In[24]:


# visualization of obtained rule
rules3.plot(kind='bar',x='support',y='confidence',color='green')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
#plt.figure(figsize=(30,10))


# In[25]:


# visualization of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[26]:


x=[0.1,0.2,0.3]
y=[52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')

