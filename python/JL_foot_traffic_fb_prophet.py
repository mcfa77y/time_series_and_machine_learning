#!/usr/bin/env python
# coding: utf-8

# # LTSM for foot traffic
# ## setup imports
# 

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow.keras.layers import Dense, Dropout, LSTM

import datetime as dt

from fbprophet import Prophet


# In[2]:


CLOSE = 'Close'
ROW_AXIS = 0
COL_AXIS = 1


# In[3]:


df = pd.read_csv('/Users/joe.lau/Downloads/bar.csv')
print(df.columns)
company = "BURGER_KING"
df = df[["Date", company]]
# # Creates the datetime object from date
df.columns=["ds", "y"]
df['ds'] = pd.to_datetime(df['ds'])
df.head()



# In[4]:


m = Prophet()
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(df)


# In[27]:


# place holder for future predictions
days_to_predict = 365
future = m.make_future_dataframe(periods=days_to_predict)

forcast = m.predict(future)


# In[21]:


forcast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(7)


# In[28]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
fig = plot_plotly(m, forcast, ylabel=f'{company} visits', xlabel='date')
fig.update_layout(title=f'{company} FB Prophet')
py.plot(fig, filename=f"{company}.html")
fig.show()


# In[29]:


m.plot_components(forcast);


# In[30]:


from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forcast)
a = add_changepoints_to_plot(fig.gca(), m, forcast)


# ## RMSE: how good is it

# In[31]:



# se = np.square(forcast.loc[:, 'yhat'][-10:] - df['y'][-10:])
se = np.square(forcast.loc[:, 'yhat'] - df['y'])
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print("rmse: " + rmse_str)


# ## RMSE: last 100 days

# In[32]:


se = np.square(forcast.loc[:, 'yhat'][-100:] - df['y'][-100:])
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print("rmse: " + rmse_str)


# In[ ]:




