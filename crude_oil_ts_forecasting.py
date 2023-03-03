#!/usr/bin/env python
# coding: utf-8

# # Crude Oil Production Forecasting

# ----------------
# ## **Context** 
# ----------------
# 
# The world economy relies heavily on hydrocarbons, particularly oil, for the provision of energy required in transportation and other industries. Crude oil production is considered one of the most important indicators of the global economy. Dependence on oil and its finite nature, pose some complex problems including estimation of future production patterns. 
# 
# Crude oil production forecasting is an important input into the decision-making process and investment scenario evaluation, which are crucial for oil-producing countries. Governments and businesses spend a lot of time and resources figuring out the production forecast that can help to identify opportunities and decide on the best way forward.
# 
# ------------------
# ## **Objective**
# ------------------
# 
# In this case study, we will analyze and use historical oil production data, from 1992 to 2018, for a country to forecast its future production. We need to build a time series forecasting model using the AR, MA, ARMA, and ARIMA models in order to forecast oil production. 
# 
# --------------------------
# ## **Data Dictionary**
# --------------------------
# 
# The dataset that we will be using is 'Crude Oil Production by Country'. This dataset contains the yearly oil production of 222 countries, but for simplicity, we will use only one country to forecast its future oil production.

# ## Importing necessary libraries

# In[38]:


# Version check 
import statsmodels

statsmodels.__version__


# In[39]:


# Libraries to do data manipulation
import numpy as np

import pandas as pd

# Library to do data visualization
import matplotlib.pyplot as plt

# Library to do time series decomposition
import statsmodels.api as sm
#from sm import tsa.seasonal_decompose

#To identify non-stationarity in the time series
from statsmodels.tsa.stattools import adfuller

# Module to create ACF and PACF plots
from statsmodels.graphics import tsaplots

# Module to build AR, MA, ARMA, and ARIMA models
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.AR.model import AutoReg

# Module to implement MSE and RSME during model evaluation
from sklearn.metrics import mean_squared_error

# Code for ignoring unnecessary warnings while executing some code  
import warnings
warnings.filterwarnings("ignore")


# This dataset has crude oil production data as time series for 222 countries starting from 1992 till 2018. This is a time series data with yearly frequency. Since the frequency of this dataset is yearly, we will not get any seasonal patterns in this time series. However, we can expect cyclicity in the data which spans over multiple years.

# **Let's load the dataset**

# In[91]:


data = pd.read_csv('Crude Oil Production by Country .csv')


data.head()


# In[94]:


data.head()


# In[102]:


x = data["1992"]
x
y = data.loc[:, data.columns != "1992"]
y


# yr = data.columns
# df = data[:1]
# df = df.T.astype("str")
# #df = pd.Series(df)
# df
# 
# #df.reset_index(yr)

# __observations__
# - there are records from 222 countries(i.e., 222 different time series)
# - We will be forecasting for only one country, i.e., `united States`

# In[41]:


# Using loc and index = 0 to fetch the data for United States from the original dataset
canada = data.loc[3]

# Dropping the variable country, as we only need the time and production information to build the model
canada = pd.DataFrame(canada).drop(['Country'])

# Fetching the two columns - YEAR and OIL PRODUCTION
canada = canada.reset_index()
canada.columns = ['YEAR', 'OIL PRODUCTION']

# Converting the data type for variable OIL PRODUCTION to integer
canada['OIL PRODUCTION'] = canada['OIL PRODUCTION'].astype(int)

# Converting the YEAR column data type to datetime
canada['YEAR'] = pd.to_datetime(canada['YEAR'])

# Setting the variable YEAR as the index of this dataframe
canada = canada.set_index('YEAR')

# Checking the time series crude oil production data for canada
canada.head()


# In[42]:


canada.isnull().sum()


# ## **Visualizing the time series and decomposing it**

# In[43]:


ax = canada.plot(color = "blue",figsize = (16,8))
ax.set_title('Yearly crude oil production by CAanada')

plt.show()


# __observations__
# 
# - The above plot shows that the oil production of Canada has been increasing gradualy  from the early 1990s till now.

# - Let's now decompose the above time series into its various components, i.e.,__trend, seasonality, and white noise__. Since this is yearly frequency data, there would not be any seasonal patterns after decomposing the time series.
# - The function, `seasonal_decompose`is obtained using moving averages. 

# In[44]:


# Using seasonal_decompose function to decompose the time series into its individual components
decomposition = sm.tsa.seasonal_decompose(canada)


# In[45]:


# Creating an empty dataframe to store the individual components
decomposed_data = pd.DataFrame()

# Extracting the trend component of time series
decomposed_data['trend'] = decomposition.trend

# Extracting the seasonal component of time series
decomposed_data['seasonal'] = decomposition.seasonal


# Extracting the white noise or residual component of time series
decomposed_data['random_noise'] = decomposition.resid


# In[46]:


#Plotting the components in a single plot

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (20, 16))

decomposed_data['trend'].plot(ax = ax1)

decomposed_data['seasonal'].plot(ax = ax2)

decomposed_data['random_noise'].plot(ax = ax3)


# __observations__
# 
# - From the above plot, the seasonal and residual components are zero, as this time series has a yearly frequency.

# ## Splitting the dataset

#  We will be Using the first 20 years data as the training data in this time series dataset

# In[47]:


# Using the first 20 years data as the training data
train_data = canada.loc['1992-01-01' : '2012-01-01']

# Using the last 7 years data as the test data
test_data = canada.loc['2012-01-01':]

train_data.shape,test_data.shape


# Now, let's visualize the train and the test data in the same plot

# In[48]:


# Creating a subplot space
fig, ax = plt.subplots(figsize = (16, 6))

# Plotting train data
train_data.plot(ax = ax)

# Plotting test data
test_data.plot(ax = ax)

# Adding the legends in sequential order
plt.legend(['train data', 'test data'])

# Showing the time which divides the original data into train and test
plt.axvline(x = '2012-01-01', color = 'black', linestyle = '--')

# Showing the plot
plt.show()


# ## Checking for stationarity

#  Before we build a time series model, we need to make sure that the time series is stationary.
# - Non-stationarity in time series may appear for the following reasons: 
# - Presence of a trend in the data
# - Presence of heteroskedasticity
# - Presence of autocorrelation

# We can identify non-stationarity in the time series by performing a statistical test called the **Augmented Dicky-Fuller Test**.
# - **Null Hypothesis:** The time series is non stationary
# - **Alternate Hypothesis:** The time series is stationary
# 

# In[49]:


# Importing ADF test from statsmodels package
from statsmodels.tsa.stattools import adfuller

# Implementing ADF test on the original time series data
result = adfuller(train_data['OIL PRODUCTION'])

# Printing the results
print(result[0]) # To get the F statistic

print(result[1]) # To get the p-value


# __observations__
# 
# -  the p-value is around 1.00, which is higher than 0.05. Hence, we fail to reject the null hypothesis, and we can say the time series is __non-stationary.__
# - We can see this visually by comparing the above ADF statistic and visually inspecting the time series.

# In[50]:


# Implementing ADF test on the original time series data
result = adfuller(train_data['OIL PRODUCTION'])

fig, ax = plt.subplots(figsize = (16, 6))

train_data.plot(ax = ax)

plt.show()

# Printing the results

print('ADF Statistic:', result[0])
print('p-value:', result[1])


# __observations__
# 
# - the time series is also seen to be  __non-stationary__ on visualiation

# We can use some of the following methods to convert a non-stationary series into a stationary one:
# 1. **Log Transformation**
# 2. **By differencing the series (lagged series)**
# 
# Let's first shift the series by order 1 (or by 1 year) and apply differencing (using lagged series).
# 
# 

# __Let now take the 1st order difference of the data and check if it becomes stationary or not__

# In[51]:


# Taking the 1st order differencing of the timeseries
train_data_stationary = train_data.diff().dropna()

# Implementing ADF test on the first order differenced time series data
result = adfuller(train_data_stationary['OIL PRODUCTION'])

fig, ax = plt.subplots(figsize = (16, 6))

train_data_stationary.plot(ax = ax)

plt.show()

# Printing the results

print('ADF Statistic:', result[0])

print('p-value:', result[1])


# __observations__
# - Here, the p-value is around 0.6911, which is again higher than 0.05. Hence, we fail to reject the null hypothesis, and we can say the time series is non-stationary.

# __Let's take the 2nd order differencing now and perform the same test.__

# In[52]:


# Taking the 2nd order differencing of the time series
train_data_stationary = train_data.diff().diff().dropna()

# Implementing ADF test on the second order differenced time series data
result = adfuller(train_data_stationary['OIL PRODUCTION'])

fig, ax = plt.subplots(figsize = (16, 6))

train_data_stationary.plot(ax = ax)

plt.show()

# Printing the results

print('ADF Statistic:', result[0])

print('p-value:', result[1])


# __observations__
# - Now, the p-value is well less than 0.05, and we can say that after taking 2nd order differencing, the time series became stationary. 
# - This parameter is also known as the **Integration** parameter (denoted by `d`) in ARIMA modeling, which we will see shortly. Here, d = 2

# In[53]:


df = (train_data)
df.head(2)
(df["OIL PRODUCTION"])


# In[54]:


df.index


# In[55]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(4, 2,figsize=(20,20),) #sharex=True)

axes[0, 0].plot(df["OIL PRODUCTION"]); axes[0, 0].set_title('Original Series')
plot_acf(df["OIL PRODUCTION"], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df["OIL PRODUCTION"].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df["OIL PRODUCTION"].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df["OIL PRODUCTION"].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df["OIL PRODUCTION"].diff().diff().dropna(), ax=axes[2, 1])


# 2nd Differencing
axes[3, 0].plot(df["OIL PRODUCTION"].diff().diff().diff()); axes[3, 0].set_title('3rd Order Differencing')
plot_acf(df["OIL PRODUCTION"].diff().diff().dropna(), ax=axes[3, 1])


plt.show()


# In[ ]:





# ## ACF and PACF Plots

# ACF and PACF plots are used to identify the model's order in ARIMA models. These plots help to find the parameters __p__ and __q__.<br> 
# Also, we always plot the ACF and PACF plots after making the time series stationary.

# **Let's generate the ACF and PACF plots.**

# In[58]:


# Creating two subplots to show ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))

# Creating and plotting the ACF charts starting from lag = 1
tsaplots.plot_acf(train_data_stationary, zero = False, ax = ax1)

# Creating and plotting the ACF charts starting from lag = 1 till lag = 8
tsaplots.plot_pacf(train_data_stationary, zero = False, ax = ax2, lags = 8)

plt.show()


# __observations__
# - From the above acf plot, it does not look like this stationary time series follows a pure MA model. As none of the plots tails off or cuts off after any lag.
# - From the above pacf plot the we observe that **the highest lag** at which the plot extends beyond the statistically significant boundary is **lag 8.**  p value is 8 for the AR model.
# - It implies that the time series follows an ARMA or ARIMA model. So, to find out the optimal values of p, d, and q, we need to do a hyper-parameter search to find their optimal values.

# Below we will try several different modeling techniques on this time series:
# - AR (p)
# - MA (q)
# - ARMA (p, q)
# - ARIMA (p, d, q)
# 
# and then we will check which one performs better i.e ,has comparable AIC to other models and less RMSE in comparison to all the other models.

# ## Evaluation Metrics

# Here, we will check the evaluation metrics - `AIC` and `RMSE`.
# - `AIC` and `RMSE` have different objectives or significance while selecting the best time series model. `RMSE` measures how far away the forecasts are in comparison to the actual values in the time series. It completely disregards the complexity of the model. Minimizing the `RMSE` provides very accurate results, but could lead to an overly complex model that captures too much noise in the data, which is also known as overfitting in the model.
# 
# - `AIC` has a different objective. `AIC` takes the error term and adds a penalty related to the number of predictors used in the model such that more complex models are penalized and allow to tradeoff between a `complex but accurate model`, against a `simpler but reasonably accurate model`.
# 
# 

# ## AR Modeling

# Below we will build the AR models at lag of 8

# In[59]:


train_data_stationary


# In[60]:


# Importing AutoReg function to apply AR model
from statsmodels.tsa.ar_model import AutoReg

plt.figure(figsize = (16, 8))

# Using number of lags as 8
model_AR = AutoReg(train_data_stationary, lags = 8)

results_AR = model_AR.fit()

plt.plot(train_data_stationary)

predict = results_AR.predict(start = 0, end = len(train_data_stationary) - 1)

# Converting NaN values to 0
predict = predict.fillna(0)

plt.plot(predict, color = 'red')

# Calculating rmse
plt.title('AR Model - RMSE: %.4f'% mean_squared_error(predict, train_data_stationary['OIL PRODUCTION'], squared = False))

plt.show()


# __observations__
# - We can see that **by using the AR model, we get root mean squared error (RMSE) = 49.3272**.
# 
# **Let's check the AIC value** of the model.

# In[61]:


# Checking the AIC value
results_AR.aic


# In[62]:


results_AR.summary()


# So the equation for this model would be:

# $$
# y_{t}= 33.9730 - 0.7406 y_{t-1} - 1.1419 y_{t-2} -1.2980 y_{t-3} - 1.0959 y_{t-4} - 1.4103 y_{t-5}- 0.8942 y_{t-6} - 1.3676 y_{t-7} - 1.1344 y_{t-8} 
# + \epsilon_{t}
# $$

# Now, let's build MA, ARMA, and ARIMA models and see if we can get a better model.

# Now, we will build several MA models at different lags and try to understand whether the MA model will be a good fit or not in comparison to the AR models that we have built so far. Below is a generalized equation for the MA model.

# $$
# y_{t}=m_{1} \epsilon_{t-1} + m_{2} \epsilon_{t-2} + \ldots+m_{q} \epsilon_{t-q} + \epsilon_{t}
# $$

# In[63]:


def plot_predicted_output(results, ax):
    
    # We are taking double cumulative sum of forecasted values (which is inverse of double differencing)
    # And we are also adding the last element of the training data to the forecasted values to get back to the original scale
    predictions = np.cumsum(np.cumsum(results.predict(start = 19, end = 25))) + train_data.iloc[-1][0]
    
    # Setting indices of the test data into prediction values
    predictions.index = test_data.index
    
    # Computing the AIC and RMSE metrics for the model and printing it into title of the plot
    train_data.plot(ax = ax, label = 'train', 
                    title = 'AIC: {}'.format(np.round(results.aic, 2)) + 
                           ' , ' +
                           'RMSE: {}'.format(np.round(np.sqrt(mean_squared_error(test_data, predictions)), 2)))
    
    # Plotting the test data
    test_data.plot(ax = ax) 
    
    # Plotting the forecasted data
    predictions.plot(ax = ax)
    
    # Adding the legends sequentially
    ax.legend(['train data', 'test data', 'forecasted values'])


# In[64]:


# We are using the ARIMA function to build the MA model, so we need to pass the stationary time series that we got after double 
# differencing the original time series. Also, we will keep the p parameter as 0 so that the model acts as an MA model

# Creating MA model with parameter q = 1
ma_1_model = ARIMA(train_data_stationary, order = (0, 0, 1))

# Creating MA model with parameter q = 2
ma_2_model = ARIMA(train_data_stationary, order = (0, 0, 2))

# Creating MA model with parameter q = 3
ma_3_model = ARIMA(train_data_stationary, order = (0, 0, 3))

# Creating MA model with parameter q = 4
ma_4_model = ARIMA(train_data_stationary, order = (0, 0, 4))


# In[65]:


# Fitting all the models that we implemented in the above cell

ma_1_results = ma_1_model.fit()

ma_2_results = ma_2_model.fit()

ma_3_results = ma_3_model.fit()

ma_4_results = ma_4_model.fit()


# In[66]:


# Plotting the forecasted values along with train and test for all the models

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 10))

plot_predicted_output(ma_1_results, ax1)

plot_predicted_output(ma_2_results, ax2)

plot_predicted_output(ma_3_results, ax3)

plot_predicted_output(ma_4_results, ax4)

plt.show()


# __observations__
# - As we can see from the above plots, again all the models that we have developed so far are comparable to AIC, but RMSE is significantly lower for MA(2) model in comparison to all the other models. So, the best model that we have got using MA modeling, is MA(2) or ARIMA(0, 0, 2). This also aligns with our observation that PACF plot seems to cut off at lag 2.  

# **Let's analyze the model summary for MA(2) or ARIMA(0, 0, 2) below.**

# In[67]:


ma_2_results.summary()


# ## **ARMA Modeling**

# From the above two models (i.e., AR and MA) that we have built so far, it looks like we have got a better model at AR(8) and on the differenced (i.e., stationary) time series data. Now, we will build several ARMA models with different combinations of p and q parameters on the differenced time series data. And we will evaluate those models based on `AIC` and `RMSE`. Let's build those models.

# Below is a generalized equation for the ARMA model.

# $$
# y_{t}=a_{1} y_{t-1}+m_{1} \epsilon_{t-1} + \ldots + \epsilon_{t}
# $$

# In[68]:


# We are using the ARIMA function here, so we need to pass stationary time series that we got after double differencing the 
# original time series

# Creating an ARMA model with parameters p = 8 and q = 1
ar_2_ma_1_model = ARIMA(train_data_stationary, order = (1, 0, 1))

# Creating an ARMA model with parameters p = 8 and q = 2
ar_2_ma_2_model = ARIMA(train_data_stationary, order=(2, 0, 2))

# Creating an ARMA model with parameters p = 8 and q = 3
ar_3_ma_2_model = ARIMA(train_data_stationary, order = (1, 0, 3))

# Creating an ARMA model with parameters p = 8 and q = 4
ar_2_ma_3_model = ARIMA(train_data_stationary, order = (1, 0, 2))


# In[69]:


# Fitting all the models that we implemented in the above cell

ar_2_ma_1_results = ar_2_ma_1_model.fit()

ar_2_ma_2_results = ar_2_ma_2_model.fit()

ar_3_ma_2_results = ar_3_ma_2_model.fit()

ar_2_ma_3_results = ar_2_ma_3_model.fit()


# In[70]:


# Plotting the forecasted values along with train and test for all the models

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 10))

plot_predicted_output(ar_2_ma_1_results, ax1)

plot_predicted_output(ar_2_ma_2_results, ax2)

plot_predicted_output(ar_3_ma_2_results, ax3)

plot_predicted_output(ar_2_ma_3_results, ax4)

plt.show()


# __observations__
# - As we can see from the above plots, again all the models that we have developed so far have comparable AIC, but for one specific model, i.e., ARIMA(2, 0, 1), the `RMSE` is significantly lower than the models that we have developed above. Also, it is evident from the above plots that the forecasted values from the model ARIMA(2, 0, 1) are closer to the test data in comparison to all the other models.

# **Let's analyze the summary for the model ARIMA(2, 0, 1).**

# In[71]:


ar_2_ma_1_results.summary()


# ## ARIMA Modeling

# While building ARIMA models, we can directly pass the non-stationary time series, as the new parameter which is required in ARIMA modeling, i.e., d parameter (along with parameters p and q) will automatically difference the data to make the time series stationary.

# In[72]:


train_data = train_data.astype('float32')


# We are using the ARIMA function here, so we do not need to pass stationary time series, we can simply pass the original time without differencing, and pass the parameter d = 2, as we already know that after double  differencing the original time series becomes a stationary time series.

# In[73]:


# Creating an ARIMA model with parameters p = 2, d = 2 and q = 1
ar_2_d_2_ma_1_model = ARIMA(train_data, order = (2, 2, 1))

# Creating an ARIMA model with parameters p = 1, d = 2 and q = 2
ar_1_d_2_ma_2_model = ARIMA(train_data, order = (1, 2, 2))

# Creating an ARIMA model with parameters p = 2, d = 2 and q = 3
ar_2_d_2_ma_2_model = ARIMA(train_data, order = (2, 2, 3))

# Creating an ARIMA model with parameters p = 3, d = 2 and q = 2
ar_3_d_2_ma_2_model = ARIMA(train_data, order = (3, 2, 2))


# In[74]:


# Fitting all the models that we implemented in the above cell

ar_2_d_2_ma_1_results = ar_2_d_2_ma_1_model.fit()

ar_1_d_2_ma_2_results = ar_1_d_2_ma_2_model.fit()

ar_2_d_2_ma_2_results = ar_2_d_2_ma_2_model.fit()

ar_3_d_2_ma_2_results = ar_3_d_2_ma_2_model.fit()


# In[75]:


def plot_predicted_output_new(results, ax):
    
    predictions = results.predict(start = 19, end = 25)
    
    # Setting indices of the test data into prediction values
    predictions.index = test_data.index
    
    # Computing the AIC and RMSE metrics for the model and printing it into title of the plot
    train_data.plot(ax = ax, label = 'train', 
                    
                    title = 'AIC: {}'.format(np.round(results.aic, 2)) + 
                           ' , ' +
                           'RMSE: {}'.format(np.round(np.sqrt(mean_squared_error(test_data, predictions)), 2)))
    
    # Plotting the test data
    test_data.plot(ax = ax) 
    
    # Plotting the forecasted data
    predictions.plot(ax = ax)
    
    # Adding the legends sequentially
    ax.legend(['train data', 'test data', 'forecasted values'])


# In[76]:


# Plotting the forecasted values along with train and test for all the models

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 10))

plot_predicted_output_new(ar_2_d_2_ma_1_results, ax1)

plot_predicted_output_new(ar_1_d_2_ma_2_results, ax2)

plot_predicted_output_new(ar_2_d_2_ma_2_results, ax3)

plot_predicted_output_new(ar_3_d_2_ma_2_results, ax4)

plt.show()


# __observations__
# - From the above analysis, we can see that the ARIMA(1, 2, 2) is the best model in comparison to others, as it has comparable AIC to other models and less RMSE in comparison to all the other models.

# **Let's analyze the model summary for ARIMA(2, 3, 2).**

# In[78]:


ar_1_d_2_ma_2_results.summary()


# **Now that we have identified the best parameters (p, d, and q) for our data. Let's train the model with the same parameters on the full data for canada and get the forecasts for the next 7 years, i.e., from 2019-01-01 to 2025-01-01.**

# In[79]:


final_model = ARIMA(canada, order = (1, 2, 2))

final_model_results = final_model.fit()


# In[80]:


forecasted_ARIMA = final_model_results.predict(start = '2019-01-01', end = '2025-01-01')


# In[82]:


# Plotting the original time seris with forecast

plt.figure(figsize = (16, 8))

plt.plot(canada, color = 'c', label = 'Original Series')

plt.plot(forecasted_ARIMA, label = 'Forecasted Series', color = 'b')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()


# - The above plot shows that the model is able to identify the trend in the data and forecast the values accordingly. 
# - The forecast indicates that, according to the historic data, the oil production is going to constantly increase for Canada

# ## Conclusion

# In[85]:


df

