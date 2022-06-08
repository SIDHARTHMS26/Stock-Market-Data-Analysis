import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title('STOCK TREND PREDICTION')
a=st.text_input("Enter The Stock Ticker ",'SBIN.NS')
data=yf.Ticker(a).history(period='10y',auto_adjust=True)

#Describing Data
st.subheader('Data of Stock Within 10 year period')
st.write(data.describe())

#Vizualization
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.xlabel('TIME')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA")
ma100=data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(data.Close,'b')
plt.xlabel('TIME')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100=data.Close.rolling(100).mean()
ma200=data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(data.Close,'b')
plt.plot(ma200,'g')
plt.xlabel('TIME')
plt.ylabel('Price')
st.pyplot(fig)

#Splitting data into training and testing dataset
data_train=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_test=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_arr=scaler.fit_transform(data_train)

#Splitting Data Into Xtrain And YTrain
x_train=[]
y_train=[]
for i in range(100,data_train_arr.shape[0]):
  x_train.append(data_train_arr[i-100:i])
  y_train.append(data_train_arr[i,0])
  
x_train,y_train=np.array(x_train),np.array(y_train)


#Machine Learning (We are using LSTM Algoritm)

from keras.layers import Dense,Dropout, LSTM
from keras.models import Sequential

model=Sequential()
model.add(LSTM(units=50, activation = 'relu', return_sequences=True,
               input_shape= (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation = 'relu', return_sequences=True,))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation = 'relu', return_sequences=True,))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

past_100_days= data_train.tail(100)
final_data=past_100_days.append(data_test, ignore_index=True)

input_data=scaler.fit_transform(final_data)

#Splitting Data Into Xtest and Ytest
x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test=np.array(x_test),np.array(y_test)

#Making Predictions
y_pred= model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor


st.subheader("PREDICTED PRICE VS ACTUAL PRICE")
fig2=plt.figure(figsize=(12,12))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_pred,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


























