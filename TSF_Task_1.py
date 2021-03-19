#!/usr/bin/env python
# coding: utf-8

# # SUPERVISED MACHINE LEARNING

# #### THE SPARK FOUNDATION ####
# #### Name: Sambit Sekhar Panda
# #### TASK1 ####

# In[2]:


# importing the essential libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# reading the data file

data= pd.read_csv("C:/Users/SamBit/Downloads/student_scores - student_scores.csv")


# In[4]:


data.head()


# In[5]:


data.describe()


# In[7]:


data.info()


# In[17]:


#Plotting the data points as a scatter plot to see the realtionship
sns.scatterplot(data['Hours'],data['Scores'],color='r')


# In[14]:


#finding out the correaltions between the two variable

data.corr()


# In[28]:


#Preparing dependent and independent Data variable

x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[29]:


#Train test split 
from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[30]:


#Importing Linear Regression Packages
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[31]:


lr.fit(x_train,y_train)
print('Model training on data is done')


# In[41]:


#Plotting the regression line on the scatter plot
rline = lr.intercept_ + lr.coef_*x_train
plt.scatter(x_train,y_train,color='b')
plt.plot(x_train,rline,color='r')
plt.title("Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")


# In[42]:


#Predicting the scores for test data
y_pred=lr.predict(x_test)
print(y_pred)


# In[51]:


#Comapring the actual values to values obtained by model
actual = list(y_test)
pred=list(y_pred)
comp = pd.DataFrame({ 'Actual':actual,'Predicted':pred})
comp


# In[57]:


#Finding Out the Errors

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
mae = metrics.mean_absolute_error(y_test, y_pred)
R2=metrics.r2_score(y_test,y_pred)
print("Mean Squared Error      = ",mse)
print("Root Mean Squared Error = ",rmse)
print("Mean Absolute Error     = ",mae)
print("R squared value     = ",R2)


# In[72]:


# Predicting Score
hours=9.25
predicted_score=lr.predict([[hours]])
print("Predicted score for a student studying ", hours ,"hours is:",predicted_score)


# In[ ]:




