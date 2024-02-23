#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing dataset

# In[5]:


data=pd.read_csv("C:\\Users\\sehga\\Downloads\\exercise.csv")
data_2=pd.read_csv("C:\\Users\\sehga\\Downloads\\calories.csv")


# ## Data Preprocessing

# In[7]:


data.head()


# In[9]:


data_2.head()


# In[10]:


data=pd.concat([data,data_2['Calories']],axis=1)


# In[11]:


data.head()


# In[12]:


data.shape


# In[13]:


data.isnull().sum()


# In[14]:


data.info()


# In[15]:


data.size


# In[16]:


data.describe()


# In[17]:


data.duplicated().sum()


# ## Data Visualization

# In[19]:


sns.distplot(x=data['Age'])


# In[20]:


sns.countplot(x='Gender', data=data)


# In[21]:


sns.distplot(x=data['Height'])


# In[22]:


age18_25=data['Age'][(data.Age<=25) &(data.Age>=18)]
age26_35=data['Age'][(data.Age<=35) &(data.Age>=26)]
age36_45=data['Age'][(data.Age<=45) &(data.Age>=36)]
age46_55=data['Age'][(data.Age<=55) &(data.Age>=46)]
age_above_55=data['Age'][(data.Age>=55)]


x=["18-25","26-35","36-45","46-55","55+ "]
y=[len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age_above_55.values)]


plt.figure(figsize=(15,6))
sns.barplot(x=x,y=y,palette="rocket")
plt.title("Number of Peoples By Age Group")
plt.xlabel("Age")
plt.ylabel("Numbers Of Peoples")
plt.show()


# In[23]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Gender',y='Height',data=data, palette='rainbow')
plt.title("Height by male, female")


# In[24]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Gender',y='Weight',data=data, palette='rainbow')
plt.title("Weight by male, female")


# In[25]:


data=data.replace({'Gender':{'male':'0','female':'1'}})


# In[26]:


data.head()


# ## Splitting dataset into training and test set

# In[29]:


x=data.iloc[:,: -1].values
y=data.iloc[:,-1].values


# In[30]:


x


# In[31]:


y


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[33]:


x_train.shape


# In[34]:


x_test.shape


# In[35]:


pip install xgboost


# ## Importing Regression Models

# In[37]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


# ## Importing metrics

# In[38]:


from sklearn.metrics import r2_score,mean_absolute_error


# ## Training Linear Regression

# In[41]:


linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred=linreg.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training Lasso Regression

# In[43]:


lasso=Lasso(alpha=0.001)
lasso.fit(x_train,y_train)
y_pred=lasso.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training Ridge Regression

# In[47]:


ridge=Ridge(alpha=0.001)
ridge.fit(x_train,y_train)
y_pred=ridge.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training Support Vector Regression

# In[50]:


svreg=SVR(kernel='rbf',C=10000,epsilon=0.1)
svreg.fit(x_train,y_train)
y_pred=svreg.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training Decision Tree Regression

# In[53]:


dectree=DecisionTreeRegressor(random_state=0)
dectree.fit(x_train,y_train)
y_pred=dectree.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training Random Forest Regressor

# In[57]:


rand=RandomForestRegressor(n_estimators=10,random_state=0)
rand.fit(x_train,y_train)
y_pred=rand.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training KNeighbors Regressor

# In[58]:


knn=KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# ## Training XGBoost Regressor

# In[59]:


xgb=XGBRegressor(n_estimators=100,max_depth=5)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)
print('R2 SCORE:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))


# In[ ]:




