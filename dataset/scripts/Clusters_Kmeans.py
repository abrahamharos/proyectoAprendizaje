#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# In[38]:


#Caso 2.1
x_train = pd.read_csv('../casos/2.1/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/2.1/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/2.1/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/2.1/YTest.csv',header=None,sep=' ')
kmeans = KMeans(n_clusters=1000)
kmeans.fit(x_train,y_train)
score_2_1 = kmeans.score(x_test,y_test)
print("Case: 2.1\n")
print("Precision: ",score_2_1)


# In[45]:


#Caso 2.2
x_train = pd.read_csv('../casos/2.2/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/2.2/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/2.2/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/2.2/YTest.csv',header=None,sep=' ')
kmeans_2_2 = KMeans(n_clusters=1000)
kmeans_2_2.fit(x_train,y_train)
score_2_2= kmeans_2_2.score(x_test,y_test)
print("Case: 2.3\n")
print("Precision: ",score_2_2)


# In[47]:


#Caso 2.3
x_train = pd.read_csv('../casos/2.3/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/2.3/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/2.3/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/2.3/YTest.csv',header=None,sep=' ')
kmeans_2_3 = KMeans(n_clusters=1000)
kmeans_2_3.fit(x_train,y_train)
score_2_3= kmeans_2_3.score(x_test,y_test)
print("Case: 2.3\n")
print("Precision: ",score_2_3)


# In[50]:


#Caso 1.1
x_train = pd.read_csv('../casos/1.1/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/1.1/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/1.1/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/1.1/YTest.csv',header=None,sep=' ')
kmeans_1_1 = KMeans(n_clusters=1000)
kmeans_1_1.fit(x_train,y_train)
score_1_1= kmeans_1_1.score(x_test,y_test)
print("Case: 1.1\n")
print("Precision: ",score_1_1)


# In[52]:


#Caso 1.2
x_train = pd.read_csv('../casos/1.2/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/1.2/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/1.2/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/1.2/YTest.csv',header=None,sep=' ')
kmeans_1_2 = KMeans(n_clusters=1000)
kmeans_1_2.fit(x_train,y_train)
score_1_2= kmeans_1_2.score(x_test,y_test)
print("Case: 1.2\n")
print("Precision: ",score_1_2)


# In[53]:


#Caso 1.3
x_train = pd.read_csv('../casos/1.3/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/1.3/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/1.3/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/1.3/YTest.csv',header=None,sep=' ')
kmeans_1_3 = KMeans(n_clusters=1000)
kmeans_1_3.fit(x_train,y_train)
score_1_3= kmeans_1_3.score(x_test,y_test)
print("Case: 1.3\n")
print("Precision: ",score_1_3)


# In[54]:


#Caso 3.1
x_train = pd.read_csv('../casos/3.1/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/3.1/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/3.1/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/3.1/YTest.csv',header=None,sep=' ')
kmeans_3_1 = KMeans(n_clusters=1000)
kmeans_3_1.fit(x_train,y_train)
score_3_1= kmeans_3_1.score(x_test,y_test)
print("Case: 3.1\n")
print("Precision: ",score_3_1)


# In[55]:


#Caso 4.1
x_train = pd.read_csv('../casos/4.1/XTrain.csv',header=None,sep=' ')
y_train = pd.read_csv('../casos/4.1/YTrain.csv',header=None,sep=' ')
x_test = pd.read_csv('../casos/4.1/XTest.csv',header=None,sep=' ')
y_test = pd.read_csv('../casos/4.1/YTest.csv',header=None,sep=' ')
kmeans_4_1 = KMeans(n_clusters=1000)
kmeans_4_1.fit(x_train,y_train)
score_4_1= kmeans_4_1.score(x_test,y_test)
print("Case: 4.1\n")
print("Precision: ",score_4_1)


# In[ ]:




