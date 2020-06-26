from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# from pyimagesearch import datasets
# from pyimagesearch import models
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import argparse
import locale
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

# -------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical

path = './total_836_200622_test.xlsx'
df = pd.read_excel(path)

## one-hot of homogeneous columns
def homogeneous(row):
    if row['Enhancement'] == 'no-enhancement':
        return 0.25
    elif row['Enhancement'] == 'weak':
        return 0.50
    elif row['Enhancement'] == 'moderate':
        return 0.75
    elif row['Enhancement'] == 'high':
        return 1
    else:
        return 0
				
## normalize ay columns				
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataset=dataNorm
    return dataset

# df_filtered = df[[u'시행시나이',u'L/R',u'cT']]
df_filtered = df[[u'시행시나이','cT']]
df_filtered['cT'] = normalize(df['cT'])
# df_filtered['L/R'] = np.where(df['L/R'] == 'L', 1, 0)
df_filtered['hu_diff'] = normalize(df['aAverage'] - df['cAverage'])
df_filtered['homogeneous'] = df.apply(homogeneous, axis=1)
df_filtered['hetero'] = np.where(df['Enhancement'] == 'hetero', 1, 0)
df_filtered['rim'] = np.where(df['Enhancement'] == 'rim', 1, 0)
df_filtered['clustered'] = np.where(df['Enhancement'] == 'clustered', 1, 0)
df_filtered['non-mass'] = np.where(df['Enhancement'] == 'non-mass', 1, 0)
df_filtered['label'] = df['pN_modify']

df_filtered = df_filtered.rename(columns={u'시행시나이': u'age'})
df_filtered.head()

properties = list(df_filtered.columns.values)
print(properties)
properties.remove('label')
print(properties)
X = df_filtered[properties]
y = df_filtered['label']
num_input = len(X.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# model = keras.Sequential([
#     Dense(6, input_shape=(7,)),
#     Activation('relu'),
#     Dense(12),
#     Activation('relu'),
#     Dense(1),
#     Activation('sigmoid')
# ])

model = Sequential()
model.add(Dense(18, activation='relu', input_dim=num_input))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(4,)),
#     keras.layers.Dense(16, activation=tf.nn.relu),
# 	keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])

model.summary()

tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit(X_train, y_train, epochs=1000, batch_size=10, callbacks=[tb_hist])
# 5. 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# TEST
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test acuracy: ', test_acc)
