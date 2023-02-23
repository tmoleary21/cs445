#!/usr/bin/env python
# coding: utf-8

# # A1.1 Linear Regression with SGD

# * A1.1: *Added preliminary grading script in last cells of notebook.*

# In this assignment, you will implement three functions `train`, `use`, and `rmse` and apply them to some weather data.
# Here are the specifications for these functions, which you must satisfy.

# `model = train(X, T, learning_rate, n_epochs, verbose)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row. $N$ is the number of samples and $D$ is the number of variable values in
# each sample.
# * `T`: is an $N$ x $K$ matrix of desired target values for each sample.  $K$ is the number of output values you want to predict for each sample.
# * `learning_rate`: is a scalar that controls the step size of each update to the weight values.
# * `n_epochs`: is the number of epochs, or passes, through all $N$ samples, to take while updating the weight values.
# * `verbose`: is True or False (default value) to control whether or not occasional text is printed to show the training progress.
# * `model`: is the returned value, which must be a dictionary with the keys `'w'`, `'Xmeans'`, `'Xstds'`, `'Tmeans'` and `'Tstds'`.
# 
# `Y = use(X, model)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row, for which you want to predict the target values.
# * `model`: is the dictionary returned by `train`.
# * `Y`: is the returned $N$ x $K$ matrix of predicted values, one for each sample in `X`.
# 
# `result = rmse(Y, T)`
# * `Y`: is an $N$ x $K$ matrix of predictions produced by `use`.
# * `T`: is the $N$ x $K$ matrix of target values.
# * `result`: is a scalar calculated as the square root of the mean of the squared differences between each sample (row) in `Y` and `T`.

# To get you started, here are the standard imports we need.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## 60 points: 40 for train, 10 for use, 10 for rmse

# Now here is a start at defining the `train`, `use`, and `rmse`
# functions.  Fill in the correct code wherever you see `. . .` with
# one or more lines of code.

# In[5]:


def train(X, T, learning_rate, n_epochs, verbose=False):

    # Calculate means and standard deviations of each column in X and T
    Xmeans = X.mean(axis=0)
    Xstds = X.std(axis=0)
    Tmeans = T.mean(axis=0)
    Tstds = T.std(axis=0)
    
    # Use the means and standard deviations to standardize X and T
    X = (X - Xmeans) / Xstds
    T = (T - Tmeans) / Tstds
    
    # Insert the column of constant 1's as a new initial column in X
    X = np.insert(X, 0, 1, axis=1)
    print(X)
    
    # Initialize weights to be a numpy array of the correct shape and all zeros values.
    w = np.zeros((X.shape[1], T.shape[1])) # D+1 x K

    for epoch in range(n_epochs):
        sqerror_sum = 0
        n_samples = X.shape[0]
        for n in range(n_samples):
            # Use current weight values to predict output for sample n, then
            # calculate the error, and
            # update the weight values.
            prediction = X[n:n+1] @ w
            error = T[n:n+1] - prediction
            w += learning_rate * X[n:n+1].T * error
            
            # Add the squared error to sqerror_sum
            sqerror_sum += error ** 2
            
        if verbose and (n_epochs < 11 or (epoch + 1) % (n_epochs // 10) == 0):
            rmse = np.sqrt(sqerror_sum / n_samples)
            rmse = rmse[0, 0]  # because rmse is 1x1 matrix
            print(f'Epoch {epoch + 1} RMSE {rmse:.2f}')

    return {'w': w, 'Xmeans': Xmeans, 'Xstds': Xstds,
            'Tmeans': Tmeans, 'Tstds': Tstds}


# In[6]:


def use(X, model):
    # Standardize X using Xmeans and Xstds in model
    X = (X - model['Xmeans']) / model['Xstds']
    # Predict output values using weights in model
    prediction = np.insert(X, 0, 1, axis=1) @ model['w']
    # Unstandardize the predicted output values using Tmeans and Tstds in model
    output = (prediction * model['Tstds']) + model['Tmeans']
    # Return the unstandardized output values
    return output


# In[7]:


def rmse(A, B):
    return np.sqrt( np.mean( (A - B)**2))


# Here is a simple example use of your functions to help you debug them.  Your functions must produce the same results.

# In[8]:


X = np.arange(0, 100).reshape(-1, 1)  # make X a 100 x 1 matrix
T = 0.5 + 0.3 * X + 0.005 * (X - 50) ** 2
plt.plot(X, T, '.')
plt.xlabel('X')
plt.ylabel('T');


# In[9]:


model = train(X, T, 0.01, 50, verbose=True)
model


# In[10]:


Y = use(X, model)
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend()


# In[11]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3)


# ## Weather Data

# Now that your functions are working, we can apply them to some real data. We will use data
# from  [CSU's CoAgMet Station Daily Data Access](http://coagmet.colostate.edu/cgi-bin/dailydata_form.pl).
# 
# You can get the data file [here](http://www.cs.colostate.edu/~cs445/notebooks/A1_data.txt)

# ## 5 points:
# 
# Read in the data into variable `df` using `pandas.read_csv` like we did in lecture notes.
# Missing values in this dataset are indicated by the string `'***'`.

# In[12]:


url = 'https://www.cs.colostate.edu/~cs445/notebooks/A1_data.txt'
df = pd.read_csv(url, delim_whitespace=True, na_values='***')
df


# ## 5 points:
# 
# Check for missing values by showing the number of NA values, as shown in lecture notes.

# In[13]:


df.isna().sum()


# ## 5 points:
# 
# If there are missing values, remove either samples or features that contain missing values. Prove that you
# were successful by counting the number of missing values now, which should be zero.

# In[14]:


df = df.dropna(axis=1) # Remove features (columns) that have missing data
df.isna().sum()


# Your job is now to create a linear model that predicts the next day's average temperature (tave) from the previous day's values. A discription of all features can be found [here](https://coagmet.colostate.edu/rawdata_docs.php). To start, consider just focusing on these features: 
# 1. tave: average temperature
# 2. tmax: maximum temperature
# 3. tmin: minimum temperature
# 4. vp: vapor pressure
# 5. rhmax: maximum relative humidity
# 6. rhmin: minimum relative humidity
# 7. pp: precipitation
# 8. gust: wind gust speed
# 
# First, modify the datafile to add a new column: 'next tave' -- here's a hint on your X and T vectors names:

# In[15]:


Xnames = ['tave', 'tmax', 'tmin', 'vp', 'rhmax', 'rhmin', 'pp', 'gust']
Tnames = ['next tave']


# __Note:__ I accomplished the desired result without adding the new column to the dataframe in the previous step. I did the manipulation in numpy instead.

# ## 5 points:
# 
# Now select those eight columns from `df` and convert the result to a `numpy` array.  (Easier than it sounds.)
# Then assign `X` to be all columns and all but the last row.  Assign `T` to be just the first column (tave) and all but the first sample.  So now the first row (sample) in `X` is associated with the first row (sample) in `T` which tave for the following day.

# In[16]:


data = df[Xnames].to_numpy()
# print(sample_data)
X = data[:-1,:]
T = data[1:,0:1]
print(X)
print(T)


# ## 15 points:
# 
# Use the function `train` to train a model for the `X`
# and `T` data.  Run it several times with different `learning_rate`
# and `n_epochs` values to produce decreasing errors. Use the `use`
# function and plots of `T` versus predicted `Y` values to show how
# well the model is working.  Type your observations of the plot and of the value of `rmse` to discuss how well the model succeeds.

# In[17]:


model = train(X, T, 0.001, 50, True)
Y = use(X, model)
# print(rmse(Y, T))
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend();


# #### Discussion
# The RMSE of our model converges quickly to 0.33, which is reasonably low. This shows that the data is a good predictor of the next day's temperature. This is also supported by the graph, which shows that the predicted values are very close in value to the actual values.

# In[18]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3);


# #### Discussion
# This graph also shows that the predicted values were very close to the target values. The points are mostly clustered very close to the red line, which shows the closeness in value. Based on the RMSE values and the two graphs, it would seem that the model was reasonably successful.

# ## 5 points:
# 
# Print the weight values in the resulting model along with their corresponding variable names (in `Xnames`). Use the relative magnitude
# of the weight values to discuss which input variables are most significant in predicting the changes in the tave values.

# In[19]:


pd.DataFrame(model['w'].T, columns=['bias', *Xnames]).sort_values(by=0, axis=1, key=lambda x: x.abs())


# #### Discussion
# The top most significant input variables are tave, tmin, and tmax. This makes sense because the temperature between two days is generally very close together, so the previous day's temperature should 

# ## Grading and Check-in
# 
# Your notebook will be partially graded automatically.  You can test this grading process yourself by downloading [A1grader.zip](https://www.cs.colostate.edu/~cs445/notebooks/A1grader.zip) and extract `A1grader.py` parallel to this notebook.  Run the code in the in the following cell to see an example grading run.  If your functions are defined correctly, you should see a score of 60/60.  The remaining 40 points are based on testing other data and your discussion.

# In[17]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# A different but similar grading script will be used to grade yout checked-in notebook.  It will include different tests.
# 
# You must name your notebook as `Lastname-A1.ipynb` with `Lastname` being your last name, and then save this notebook and check it in at the A1 assignment link in our Canvas web page.

# ## Extra Credit: 1 point

# A typical problem when predicting the next value in a time series is
# that the best solution may be to predict the previous value.  The
# predicted value will look a lot like the input tave value shifted on
# time step later.
# 
# To do better, try predicting the change in tave from one day to the next. `T` can be assigned as

# In[18]:


T = data[1:, 0:1] -  data[:-1, 0:1]


# Now repeat the training experiments to pick good `learning_rate` and
# `n_epochs`.  Use predicted values to produce next day tave values by
# adding the predicted values to the previous day's tave.  Use `rmse`
# to determine if this way of predicting next tave is better than
# directly predicting tave.

# In[19]:


model = train(X, T, 0.001, 50, True)
Y = use(X, model)
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend();


# In[20]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3);


# In[ ]:




