#!/usr/bin/env python
# coding: utf-8

# In[1]:


# API lib

import requests
import base64
import json
from math import sqrt
import numpy as np
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


# In[2]:


# Test GPU
from tensorflow.python.client import device_lib
import tensorflow as tf
print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())


# ### Init

# In[3]:


# APIs
api_key = ''
api_header = {}
with open('api_header.json') as f:
    api_header = json.load(f)
f = open("api_key.txt", "r")
api_key = f.read()
api_header["X-IG-API-KEY"] = api_key

# from Crypto.PublicKey import RSA
# from Crypto.Cipher import PKCS1_v1_5

# login_url = "https://demo-api.ig.com/gateway/deal/"

# data=json.dumps({
# 'encryptedPassword': False,
# 'identifier': identifier,
# 'password': password
# })


# x = requests.post(login_url, data = data)
# print(x.text)


# m_data = r.json()

# decoded = base64.b64decode(m_data['encryptionKey'])
# rsakey = RSA.importKey(decoded)
# message = password + '|' + str(long(m_data['timeStamp']))
# input = base64.b64encode(message)
# encryptedPassword = base64.b64encode(PKCS1_v1_5.new(rsakey).encrypt(input))

# session = "/session"
# m_url = url + session
# headers = { "Content-Type": "application/json; charset=utf-8",
# "Accept": "application/json; charset=utf-8",
# "X-IG-API-KEY": m_apiKey,
# "Version": "2"
# }

# payload = json.dumps({ "identifier": identifier,
# "password": encryptedPassword,
# "encryptedPassword": True
# })

# # In[]
# r = requests.post(m_url, data=payload, headers=headers)
# r.status_code
# print r.status_code
# print r.text

pd.options.display.max_columns = None


# start_session_init()
date0 = '2019-05-01T00%3A00%3A00'
date1 = '2019-08-01T00%3A00%3A00'
date2 = '2020-04-22T23%3A59%3A59'
# resolution = 'HOUR_2'
resolution = 'MINUTE_30'
xau_epic = 'CS.D.CFDGOLD.CFDGC.IP'
usd_epic = 'CO.D.DX.FWS2.IP'
us500_epic = 'IX.D.SPTRD.IFD.IP'
us100_epic = 'IX.D.NASDAQ.IFD.IP'
eur_epic = 'CS.D.EURUSD.CFD.IP'
ftse_epic = 'IX.D.FTSE.CFD.IP'
eurchn_epic = 'CS.D.EURCNH.CFD.IP'
usdchn_epic = 'CS.D.USDCNH.CFD.IP'
usoil_epic = 'CC.D.CL.UNC.IP'


# Functions

# In[4]:


def start_session():
    url = "https://demo-api.ig.com/gateway/deal"
    session = "/session/encryptionKey"
    m_url = url + session    
    return  requests.get(m_url, headers=headers)

def price_history(epic,resolution,date1,date2):
    m_url = "https://api.ig.com/gateway/deal/prices/{}?resolution={}&from={}&to={}&pageSize=0".format(epic,resolution,date1,date2)
     # "Version": "2"
    return  requests.get(m_url, headers=headers)

def price_extractor(dfx,obj):
    # always have problem with str or not str. please convert to csv first
    suffix = '' # obj + '_' 
    dfx[suffix +'openPrice'] = dfx['openPrice'].apply(lambda x: (eval(x)).get('ask'))
    dfx[suffix +'closePrice'] = dfx['closePrice'].apply(lambda x: (eval(x)).get('ask'))
    dfx[suffix +'highPrice'] = dfx['highPrice'].apply(lambda x: (eval(x)).get('ask'))
    dfx[suffix +'lowPrice'] = dfx['lowPrice'].apply(lambda x: (eval(x)).get('ask'))

def ts(new_data, look_back = 100, pred_col = 1):
    t = new_data.copy()
    t['id'] = range(1, len(t)+1)
    t = t.iloc[:-look_back,:]
    t.set_index('id', inplace= True)
    pred_value = new_data.copy()
    pred_value = pred_value.iloc[look_back:, pred_col]
    pred_value.columns = ['Pred']
    pred_value = pd.DataFrame(pred_value)
    pred_value['id'] = range(1,len(pred_value)+1)
    pred_value.set_index('id', inplace= True)
    final_df= pd.concat([t,pred_value],axis=1)
    return final_df



# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def adj_r2(x,r2):
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[31]:


archive_path = 'data/'
df_prices = pd.read_csv('gold_feature_price'+ '_' + resolution +'.csv', parse_dates=['snapshotTime']) # Data Restore


# In[46]:


# Feature Selection
features_price = df_prices.copy()

datetime_price = features_price[['snapshotTime']]
features_price = features_price[['closePrice','closePrice_usd','closePrice_eur','closePrice_usoil','closePrice_us500','closePrice_us100','closePrice_ftse']] #,'closePrice_eurchn']
# features_price = features_price[['price_change','price_change_usd','price_change_us500','price_change_usoil', 'price_change_eur']] 

features_price.plot(subplots=True)
plt.show()


# In[47]:


# Split test/training
counter = features_price.shape[0]
prediction_period = 48*3
split_point1 = counter - prediction_period
split_point2 = 7600
features_matrix = features_price.values.astype('float32')
train_price = features_matrix[:split_point1,]
test_price = features_matrix[split_point1:,]
datetime_price = datetime_price.iloc[split_point1:,]
# test_price = features_matrix[split_point2:,]
scaler, train_scaled, test_scaled = scale(train_price,test_price)

print(train_price.shape, test_price.shape, datetime_price.shape)


# ### RNN Matrix prep

# In[48]:


# The Scale
rnn_time_steps = 48
n_features = train_price.shape[1]
print(n_features)
scaler, train_scaled, test_scaled = scale(train_price,test_price)
print(train_scaled.shape)


# Matrix Reformation methods

# In[49]:


# One feature method
# gold_price_scaled = scaled_price #[:,0:1]
# rnn_size = 300
# X_train = list()
# y_train = list()
# for i in range(rnn_size, spliting_point):
#     X_train.append(gold_price_scaled[i-rnn_size:i, 0])
#     y_train.append(gold_price_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)

# series_to_supervised(scaled, n_hours, 1)
train_scaled = series_to_supervised(train_scaled,n_in=rnn_time_steps,n_out=1)
train_scaled = train_scaled.values
test_scaled = series_to_supervised(test_scaled,n_in=rnn_time_steps,n_out=1)
test_scaled = test_scaled.values

train_scaled.shape


# In[50]:


# Split into Input and output, X & Y

n_obs = rnn_time_steps * n_features
x_train, y_train = train_scaled[:, :n_obs], train_scaled[:, -n_features]
x_test, y_test = test_scaled[:, :n_obs], test_scaled[:, -n_features]

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[51]:


# # Reshaping (batch_size, timesteps, input_dim) # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

X_train = x_train.reshape(x_train.shape[0], rnn_time_steps, n_features)
X_test = x_test.reshape(x_test.shape[0], rnn_time_steps, n_features)
print(X_train.shape,X_test.shape)


# #### LSTM Model

# In[ ]:


#  RNN Model libs
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Init RNN
regressor = Sequential()

# 1st LSTM Layer + dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# 2nd LSTM Layer + dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# # 3rd LSTM Layer + dropout regularisation
# regressor.add(LSTM(units = 20, return_sequences = True))
# regressor.add(Dropout(0.2))

# 4th LSTM Layer+ dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compile the model
regressor.compile(  optimizer = 'adam', loss = 'mean_squared_error') #, metrics = ['accuracy'])

# Fitting the RNN to the training set
regressor.fit(X_train,y_train, epochs = 30, batch_size=48, validation_data=(X_test, y_test), verbose=2, shuffle=False)
regressor.save('reg_lstm_v1_f7_1.h5')


# In[35]:


# Restore a Model to aviod test set conflicts
# regressor.save('reg_lstm4.h5')
regressor = load_model('reg_lstm_v1_f7_1.h5')


# ### Test the Model

# In[19]:


# Test the RNN
# dataset_test = test_price #['closePrice']
# real_gold_price = dataset_test.iloc[:,0:1].values

# # dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# dataset_total = features_price #['closePrice']
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 200:].values
# inputs = inputs.reshape(-1,1)
# gold_test_scaled = sc.transform(inputs[:,0:1])
# # gold_test_scaled = 

# # RNN Data Structure
# X_test = list()
# for i in range(200, 277):
#     X_test.append(gold_test_scaled[i-200:i, 0])
# X_test = np.array(X_test)

# # Reshaping
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction
y_pred = regressor.predict(X_test)
# Hint: test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
X_test = X_test.reshape((X_test.shape[0], rnn_time_steps*n_features))
# invert scaling for forecast
inv_yhat = concatenate((y_pred, X_test[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, -(n_features-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# predicted_gold_price = invert_scale(scaler, X_test, y_pred)
# predicted_gold_price = scaler.inverse_transform(predicted_gold_price)
# predicted_gold_price = predicted_gold_price[:,0]
# predicted_gold_price = predicted_gold_price.reshape(-1,1)


# In[20]:


# calculate RMSE
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print(inv_y.shape, inv_yhat.shape)
print('Test RMSE: %.3f' % rmse)



# In[106]:


# Visualization the results
plot_size1 = None # Max 165
plot_size2 = None
# ax = plt.axes()
fig, ax = plt.subplots(figsize=(15,6))
# ax.plot(t, s)

# fig = plt.figure(figsize=(15,6))
ax.plot(inv_y[plot_size1:plot_size2],datetime_price, color= 'red', label = 'Real Gold Price')
ax.plot(inv_yhat[plot_size1:plot_size2], color= 'blue', label = 'Predicted Gold Price')
ax.set_title('Gold Price Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
ax.xaxis.set_major_locator(MultipleLocator(12))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which='both');
plt.savefig('gold_price_prediction_.png' , dpi=150)
plt.show()


# In[ ]:


y_pred


# In[40]:


X_test.shape, y_pred.shape

