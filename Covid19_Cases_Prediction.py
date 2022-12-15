#%%
#packages
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import datetime
import os

# %%
# 1. Data Loading

CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
df_train = pd.read_csv(CSV_PATH)

#%%
#2. Data Inspection

df_train.info()

df_train.describe()

# %%
# 3. Data Cleaning

# Check for completed duplicated from complete dataframe
df_train.duplicated().sum()

# Extract a column as feature
cases = df_train['cases_new']

# Check for missing values 
print(f'Missing values : {cases.isna().sum()}')  # no missing values but have characters elements

# Change character values to NaN
cases = pd.to_numeric(cases,errors='coerce')

# Check again for missing values 
print(f'Missing values : {cases.isna().sum()}') # 12 missing values

# Plot graph to see missing values
plt.figure(figsize=(10,10))                         
plt.plot(cases)
plt.show()

# Fill NaN with number using interpolate 
cases = cases.interpolate(method='polynomial',order=2)

# # Plot graph again to see fill values
plt.figure(figsize=(10,10))                         
plt.plot(cases)
plt.show()

# %%
# 5. Data preprocessing

# reshape cases into 2D
cases = cases[::,None]

# Normalize feature
mm_scl = MinMaxScaler()
cases = mm_scl.fit_transform(cases)

# Create 30 days window
win_size = 30
X_train = []
Y_train = []

# loop 30 days window and append list
for i in range (win_size,len(cases)):
    X_train.append(cases[i-win_size:i])
    Y_train.append(cases[i])

# Convert list to array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Train test split features
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,random_state=12345)

# %%
# 6. Model development

model = Sequential()
model.add(LSTM(64,input_shape= x_train.shape[1:],return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

model.summary()
# %%
# 7. Model compilation and training

# saved lod path destination
log_path = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%M%d-%H%M%S'))

# tensorboard callbacks
tb = TensorBoard(log_dir=log_path)
es = EarlyStopping(monitor='val_loss',patience=5)

# model compilation
model.compile(optimizer='adam',loss='mse',metrics=['mape'])

# model training
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,callbacks=[tb,es])

# %%
# 8. Model evaluation

# csv test path
CSV_TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

# load csv test
cases_test = pd.read_csv(CSV_TEST_PATH) 

# Check for duplicates and missing values
print(f'Completed duplicates :{cases_test.duplicated().sum()}')
print(f'Missing values :\n{cases_test.isna().sum()}')

# Extract target from test dataframe
cases_test = cases_test['cases_new']
 
# Fill NaN with number using interpolate 
cases_test = cases_test.interpolate(method='polynomial',order=2)

# reshape cases into 2D
cases_test = cases_test[::,None]

#convert cases and cases_test to dataframe
cases = pd.DataFrame(cases)
cases_test = pd.DataFrame(cases_test)

# concat cases with cases_test
concat = pd.concat((cases,cases_test),ignore_index=True,axis=0)

# slice concat
concat = concat[len(concat)-win_size-len(cases_test):]

# normalize concat
concat = mm_scl.fit_transform(concat)

# empty list
x_test_list = []
y_test_list = []

# loop window and append list
for i in range (win_size,len(concat)):
    x_test_list.append(concat[i-win_size:i])
    y_test_list.append(concat[i])

# convert to array
X_test_ = np.array(x_test_list)
Y_test_ = np.array(y_test_list)

# %%
# 9. Model Prediction

predicted_test = model.predict(X_test_)

predicted_test = mm_scl.inverse_transform(predicted_test)
Y_test_ = mm_scl.inverse_transform(Y_test_)

plt.figure(2)
plt.plot(predicted_test,color='r')
plt.plot(Y_test_,color='b')
plt.legend(['Predicted Cases','Actual Cases'])
plt.xlabel('time')
plt.ylabel('Covid19 cases')
plt.show()

#metrics to evaluate the performance
print('MAPE :',mean_absolute_percentage_error(Y_test_,predicted_test))
print('MSE :',mean_squared_error(Y_test_,predicted_test))

# %%
# 10. Model saving

# saving normalization parameter
with open('mm_scl.pkl','wb') as f:
    pickle.dump(mm_scl,f)

# saving entir model
model.save('Covid19_Pred.h5')
# %%
