#!/usr/bin/env python
# coding: utf-8

# # LTSM for foot traffic
# ## setup imports
# 

# In[3]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Dense, Dropout, LSTM

import datetime as dt

import plotly
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.express as px
import plotly.graph_objects as go


# # Utils

# In[4]:


def list_to_np_array(tmp_list, label=""):
    result = np.array(tmp_list)
    result = np.reshape(result, (result.shape[0], result.shape[1], 1))
    print(f"{label}: row: {result.shape[0]} col: {result.shape[1]}")
    return result


# ### constants

# In[5]:


#  call these brands since they are agregated info
company = 'BURGER_KING'
ROW_AXIS = 0
COL_AXIS = 1


# ## Load Data
# 

# In[6]:


start = dt.datetime(2020, 4, 1)
end = dt.datetime(2020, 12, 31)
# end = dt.datetime.now() - dt.timedelta(days=1)
data = pd.read_csv('/Users/joe.lau/Downloads/bar.csv',index_col = 'Date')


# Creates the datetime object 
data.index = pd.to_datetime(data.index)

## Gets the training data from the start to the first month of January
train_data = data.loc[start:end].copy(deep = True)

# train_data = web.DataReader(company, 'yahoo', start, end)
train_company_data = train_data[company]

# train_data_close = train_data[CLOSE].values.reshape(-1,1)


# In[7]:


data


# ## Prepare Data

# In[8]:


# scale date to fit between 0 - 1
scalar = MinMaxScaler(feature_range=(0,1))
scaled_train_data = scalar.fit_transform(train_company_data.values.reshape(-1,1))
print(f"scaled_train_data.shape: {scaled_train_data.shape}")

# how many days to look into past
# window of data to use for pridiction
# larger window is less sensative to recent stock prices
# smaller window is more sensitive to recent stock prices
prediction_days = 5

size_data = len(scaled_train_data)

# x_train is a sliding window of 60 days of data
x_train = []

# y_train is the closing day value of stocks on 60th day
y_train = []

for x in range(prediction_days, size_data):
    prev_x = x - prediction_days
    x_training_data = scaled_train_data[prev_x:x, 0]
    # x_training_data shape = (60,)
    x_train.append(x_training_data)

    y_training_data = scaled_train_data[x, 0]
    # y_training_data shape = ()
    y_train.append(y_training_data)

x_train = list_to_np_array(x_train, "x_train")
y_train = np.array(y_train)


# ## Build model

# In[9]:


model = Sequential()
# you can tweak this parameter
units = 15
# x_train.shape[1] = 30
model.add(LSTM(units=units, return_sequences= True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=units, return_sequences= True))
model.add(Dropout(0.2))

model.add(LSTM(units=units))
model.add(Dropout(0.2))

model.add(Dense(units=1)) # predict the closing price

model.compile(optimizer="adam", loss="mean_squared_error", metrics=[RootMeanSquaredError(name="rmse")])
model.fit(x_train, y_train, epochs=25, batch_size=32)


# ## Load test data

# In[10]:


test_start = dt.datetime(2020,1,1)
test_end = dt.datetime(2021,1,17)

test_data = data.loc[test_start:test_end].copy(deep = True)
# test_data = web.DataReader(company, 'yahoo', test_start, test_end)
y_test = test_data[company].values
print(f"train_data[company].shape: {train_data[company].shape}")
print(f"test_data[company].shape: {test_data[company].shape}")

total_dataset = pd.concat((train_data[company], test_data[company]), axis=ROW_AXIS)
print("total_dataset.shape", total_dataset.shape, type(total_dataset))
start_index = len(total_dataset) - len(test_data) - prediction_days
print(f"start_index: {start_index}")

# model_inputs is a list
model_inputs = total_dataset[start_index:].values
model_inputs_date = total_dataset[start_index:].index
# print("total_dataset[start_index:].values", model_inputs.shape, model_inputs[:-1])

# model_inputs is a column
model_inputs = model_inputs.reshape(-1,1)
# print("model_inputs.reshape(-1,1)", model_inputs.shape, model_inputs[:-1])

# model_input is a column of scaled values
model_inputs = scalar.transform(model_inputs)
# print("scalar.transform(model_inputs)", model_inputs.shape, model_inputs[:-1])


# ## Make predictions on test data

# In[11]:


x_test = []
x_test_date = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    x_test_date.append(model_inputs_date[x])

x_test = list_to_np_array(x_test, "x_test")

predicted_prices = model.predict(x_test)
# reverse the scaling
predicted_prices = scalar.inverse_transform(predicted_prices)


# In[12]:


np.shape(x_test_date)


# ## Plot test predictions

# In[13]:


predicted_prices_length = predicted_prices.size
fig = go.Figure()
x_range = np.linspace(0, predicted_prices_length, predicted_prices_length)

fig.add_trace(go.Scatter(name="predicted", x=x_test_date, y=predicted_prices.reshape(predicted_prices_length)))
fig.add_trace(go.Scatter(name="actual", x=x_test_date, y=y_test.reshape(predicted_prices_length)))
fig.update_layout(title=f'{company} (LSTM)', xaxis_title="time", yaxis_title=f"{company} price")
fig.show()


# ## Predict next day

# In[14]:


row_selection = len(model_inputs) + 1 - prediction_days
col_selection = len(model_inputs + 1)
real_data = [model_inputs[row_selection: col_selection, 0]]
real_data = list_to_np_array(real_data, "real_data")

prediction = model.predict(real_data)
prediction = scalar.inverse_transform(prediction)[0][0]

prediction_str = "{:0.2f}".format(prediction)
test_end_str = test_end.strftime("%m-%d-%Y")
print(f"Prediction for {test_end_str}:  visits {prediction_str}")


# ## RMSE how good is it?

# In[15]:


se = np.square(y_test - predicted_prices)
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print(f'RMSE: {rmse_str}')

# update rmse calc for looking at the last 100 days 


# ## RMSE: last 100 days

# In[16]:


se = np.square(y_test[-100:] - predicted_prices[-100:])
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print("rmse: " + rmse_str)


# In[ ]:




