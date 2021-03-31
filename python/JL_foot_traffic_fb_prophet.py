#!/usr/bin/env python
# coding: utf-8

# # LTSM for foot traffic
# ## setup imports
# 

# In[41]:


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


# In[42]:


CLOSE = 'Close'
ROW_AXIS = 0
COL_AXIS = 1


# In[43]:


df = pd.read_csv('/Users/joe.lau/Downloads/bar.csv')
print(df.columns)
company = "BURGER_KING"
df = df[["Date", company]]
# # Creates the datetime object from date
df.columns=["ds", "y"]
df['ds'] = pd.to_datetime(df['ds'])
df.head()



# In[44]:


m = Prophet()
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(df)


# In[45]:


# place holder for future predictions
days_to_predict = 365
future = m.make_future_dataframe(periods=days_to_predict)

forcast = m.predict(future)


# In[46]:


forcast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(7)


# In[47]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
fig = plot_plotly(m, forcast, ylabel=f'{company} visits', xlabel='date')
fig.update_layout(title=f'{company} FB Prophet')
py.plot(fig, filename=f"{company}.html")
fig.show()


# In[48]:


m.plot_components(forcast);


# In[49]:


from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forcast)
a = add_changepoints_to_plot(fig.gca(), m, forcast)


# ## RMSE: how good is it

# In[50]:



# se = np.square(forcast.loc[:, 'yhat'][-10:] - df['y'][-10:])
se = np.square(forcast.loc[:, 'yhat'] - df['y'])
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print("rmse: " + rmse_str)


# ## RMSE: last 100 days

# In[68]:


se = np.square(forcast.loc[:, 'yhat'][-100:].values - df['y'][-100:].values)
mse = np.mean(se)
rmse = np.sqrt(mse)
rmse_str = "{:0.2f}".format(rmse)
print("rmse: " + rmse_str)

