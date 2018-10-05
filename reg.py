
# coding: utf-8

# # Midterm Practice: Predicting Boston Home Values
# 
# In this lab, we are predicting the natural log of the sum of all transactions per user. This is a great chance to practice all of our skills to date in order to create a regression model.
#   
# # Variable Descriptions
# 
# This data frame contains the following columns:
# 
# #### crim  
# per capita crime rate by town.
# 
# #### zn  
# proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# #### indus  
# proportion of non-retail business acres per town.
# 
# #### chas  
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 
# #### nox  
# nitrogen oxides concentration (parts per 10 million).
# 
# #### rm  
# average number of rooms per dwelling.
# 
# #### age  
# proportion of owner-occupied units built prior to 1940.
# 
# #### dis  
# weighted mean of distances to five Boston employment centres.
# 
# #### rad  
# index of accessibility to radial highways.
# 
# #### tax  
# full-value property-tax rate per $10,000.
# 
# #### ptratio  
# pupil-teacher ratio by town.
# 
# #### black  
# 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# 
# #### lstat  
# lower status of the population (percent).
# 
# #### medv  
# median value of owner-occupied homes in $10000s.
#   
#   
#   
# Source
# Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.
# 
# Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley.

# # Import Data

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


# In[27]:


df = pd.read_csv('train.csv')
print(len(df))
df.head()


# # Define Variables, Create an Initial Model and Measuring Model Performance

# In[44]:


X = df.drop('medv', axis=1)
y = df.medv
X_train, X_test, y_train, y_test = train_test_split(X, y)
models = [LinearRegression(), Lasso(), Ridge()]
names = ['OLS', 'Lasso', 'Ridge']
for model, name in list(zip(models, names)):
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test) 
    print('Model Stats for: {}'.format(name))
    print('Train R^2:', model.score(X_train, y_train))
    print('Test R^2:', model.score(X_test, y_test))
    print('Training MSE: {}'.format(mean_squared_error(y_train, y_hat_train)))
    print('Testing MSE: {}'.format(mean_squared_error(y_test, y_hat_test)))
    print('\n')    


# # Using Cross Validation

# In[45]:


model = LassoCV()
model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test) 
print('Model Stats for: {}'.format('LassoCV'))
print('Train R^2:', model.score(X_train, y_train))
print('Test R^2:', model.score(X_test, y_test))
print('Training MSE: {}'.format(mean_squared_error(y_train, y_hat_train)))
print('Testing MSE: {}'.format(mean_squared_error(y_test, y_hat_test)))
print('Model details:', model)
print('\n')    


# In[46]:


model.alpha_


# In[47]:


model = RidgeCV()
model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test) 
print('Model Stats for: {}'.format('RidgeCV'))
print('Train R^2:', model.score(X_train, y_train))
print('Test R^2:', model.score(X_test, y_test))
print('Training MSE: {}'.format(mean_squared_error(y_train, y_hat_train)))
print('Testing MSE: {}'.format(mean_squared_error(y_test, y_hat_test)))
print('Model details:', model)
print('\n')    


# In[48]:


model.alpha_


# # Feature Engineering + Refinements

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[49]:


#Fill Null Values and Normalize
for col in X.columns:
    avg = X[col].mean()
    X[col] = X[col].fillna(value=avg)
    minimum = X[col].min()
    maximum = X[col].max()
    range_ = maximum - minimum
    X[col] = X[col].map(lambda x: (x-minimum)/range_)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Make a pipeline model with polynomial transformation
#Currently with basic ridge.
#Could use and LASSO regression with cross-validation, (included in comments)
degree_min = 2
degree_max=5

degrees = []
train_errs = []
test_errs = []
for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),
                          Ridge()
                         )
    #Could replace Ridge() above with a more complicated cross validation method to improve tuning
    #using a cross validation method will substantially increase runtime
    model.fit(X_train,y_train)
    #Get r^2 values for testing predictions and training predictions
    test_score = model.score(X_test,y_test)
    test_errs.append(test_score)
    
    train_score = model.score(X_train,y_train)
    train_errs.append(train_score)
    
    degrees.append(degree)
#Create Plot
plt.scatter(degrees, train_errs, label='Train R^2')
plt.scatter(degrees, test_errs, label='Test R^2')
plt.title('Train and Test Accuracy vs Model Complexity')
plt.xlabel('Maximum Degree of Polynomial Regression')
plt.legend()


# # Comment:
# 
# While training accuracy continues to improve with model complexity, we see diminished returns after degree 3 leading us to believe the model is overfit past that point. As such, we will try and finalize our model using a polynomial of degree3.

# In[51]:


model = make_pipeline(PolynomialFeatures(3, interaction_only=False),
                          Ridge()
                         )
#Could replace Ridge() above with a more complicated cross validation method to improve tuning
#using a cross validation method will substantially increase runtime
model.fit(X_train,y_train)
test_score = model.score(X_test,y_test)
print('R^2 Test:', test_score)
train_score = model.score(X_train,y_train)
print('R^2 Train:', train_score)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test) 
print('Model Stats for: {}'.format('LassoCV'))
print('Training MSE: {}'.format(mean_squared_error(y_train, y_hat_train)))
print('Testing MSE: {}'.format(mean_squared_error(y_test, y_hat_test)))
print('Model details:', model)
print('\n')    


# # Additional Notes / Comments
# Much more work could still be done. Building a model is an ongoing process of refinement and is often terminated early due to time constraints or satisfaction with the model results. 
# 
# https://www.kaggle.com/c/boston-housing
