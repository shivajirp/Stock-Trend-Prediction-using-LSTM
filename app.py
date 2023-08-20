import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import datetime as dt
# ismport pandas_datareader as data

from keras.models import load_model
import streamlit as st



st.title('Stock Trend Prediction')

stock_name = st.text_input("Enter Stock Ticker", 'AAPL')
start = st.text_input("Start Date")
end = st.text_input("End Date")

# start = '2010-01-01'
# end = '2022-01-01'

df = yf.download(stock_name, start, end)

# Describing Data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#  VISUALISATIONS


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma200 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


# MOVING ON TO MODEL

# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7) : int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# i have already so i will directily use it

# Loading model
model = load_model('second_model.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)


y_pred = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
