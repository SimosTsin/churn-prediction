# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:01:04 2019

@author: simos
"""


import pandas as pd

# importing the data and preprocessing them
dataset ="path\\train.csv"
dataset = pd.read_csv(dataset)
dataset = dataset.drop(['state','area_code'], axis=1)
dataset.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
dataset.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)
dataset.churn.replace(('yes', 'no'), (1, 0), inplace=True)

# Test dataset
pred_data ="path\\test.csv"
pred_data = pd.read_csv(pred_data)
pred_data = pred_data.drop(['state','area_code','id'], axis=1)
pred_data.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
pred_data.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)



# A usefull insight
dataset.shape
dataset.head()
dataset.groupby('churn').size() # pretty unbalanced data

# Correlation matrix
import matplotlib.pyplot as plt
plt.matshow(dataset.corr())
plt.title('Correlation matrix')
dataset = dataset.drop(['total_day_minutes','total_eve_minutes','total_night_minutes','total_intl_minutes'], axis=1)
pred_data = pred_data.drop(['total_day_minutes','total_eve_minutes','total_night_minutes','total_intl_minutes'], axis=1)


dataset = dataset.values 
pred_data = pred_data.values
data = dataset[:,0:13] 
target = dataset[:,13]  # churn attribute

# Splitting the data into the training set and test set
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2)


# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
data = scaler.transform(data)
pred_data = scaler.transform(pred_data)


# Building the model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


classifier = Sequential()
classifier.add(Dense(22, input_dim = 13, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(27, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(32, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(37, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(43, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(50, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(57, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(51, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(45, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(39, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(33, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(27, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(21, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(15, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(9, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
classifier.fit(data_train, target_train, batch_size = 20, epochs = 200)


# Prediction on test data
y_pred = classifier.predict(data_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test, y_pred)

tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
count_all = tp+tn+fp+fn

accuracy = (tp+tn)/count_all
error_rate = (fp+fn)/count_all
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f_score = (precision*recall)/((precision+recall)/2)
print('Accuracy =',accuracy*100,'%','and F-score =',f_score*100,'%')


# The actual prediction
prediction = classifier.predict(pred_data)
prediction = (prediction > 0.5)
column_new = ['churn']
prediction=pd.DataFrame(prediction, columns=column_new)
prediction.replace((1,0),('yes', 'no'), inplace=True)
column_new=['id']
id=pd.DataFrame([i for i in range (1,751)],columns=column_new)
prediction=pd.concat([id,prediction],axis=1)
prediction.to_csv("path\\save_name.csv",index=False)
