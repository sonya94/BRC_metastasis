#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name : 6.NN_RFE_20210726.py
@Author : Soyoung Park
@Contect : thegiggles@naver.com
@Time    : 2021-07-26 오후 7:10
@Desc: Run NN with RFE (lVolume included)
"""

import warnings

warnings.filterwarnings(action='ignore')
import os, sys
import pandas as pd
import numpy as np

def importOwnLib():
    if '/mnt/8TBDisk/github/lib' not in sys.path:
        sys.path.append('/mnt/8TBDisk/github/lib')
        print("lib path is successfully appended.")
    else:
        print("lib path is already exists.")

importOwnLib()
import sonyalib as sonya

import importlib
importlib.reload(sonya)
from math import sqrt
import datetime
import timeit
import tensorflow as tf

from scipy import interp

import keras
from keras.optimizers import Adam
from keras import callbacks, losses

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras import initializers
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
## RFE
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, roc_curve


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # GPU메모리가 전부 할당되지 않고, 아주 적은 비율만 할당되어 시작해서 프로세스의 메모리 수요에 따라 증가하게 된다.
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

def mlp_model(num_input, dropout=0, lr=0.005, l1=9, l2=9):
    keras.backend.clear_session()
    seed_number = 7
    ## 모델 구성하기
    model = Sequential()
    # print learning rate

    model.add(Dense(l1, activation='relu', input_dim=num_input, kernel_initializer=initializers.he_normal(seed=seed_number)))
    model.add(BatchNormalization())
    model.add(Dense(l2, activation='relu', kernel_initializer=initializers.he_normal(seed=seed_number)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.he_normal(seed=seed_number)))

    ## 모델 컴파일
    model.compile(optimizer=Adam(lr), loss=losses.binary_crossentropy, metrics=['accuracy'])

    # model.summary()
    return model


def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))

    # calculate mis-classification
    conf_misclassification = 1 - conf_accuracy

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-' * 50)
    print(f'Accuracy: {round(conf_accuracy, 2)}')
    print(f'Mis-Classification: {round(conf_misclassification, 2)}')
    print(f'Sensitivity: {round(conf_sensitivity, 2)}')
    print(f'Specificity: {round(conf_specificity, 2)}')
    print(f'Precision: {round(conf_precision, 2)}')
    print(f'f_1 Score: {round(conf_f1, 2)}')

    return conf_sensitivity,  conf_specificity


def load_excel(nPath):
    return pd.read_excel(nPath, sheet_name='BRC_metastasis_input', dtype={u'ID': str})

def modify_homogeneous(row):
    if row['enhancement'] == 'non-enhancement':
        return 0.25
    elif row['enhancement'] == 'weak':
        return 0.50
    elif row['enhancement'] == 'moderate':
        return 0.75
    elif row['enhancement'] == 'high':
        return 1
    else:
        return 0


def modify_cN(row):
    if row['cN'] == 'suspicious':
        return 0.25
    elif row['cN'] == 'possible':
        return 0.50
    elif row['cN'] == 'probable':
        return 0.75
    elif row['cN'] == 'definite':
        return 1
    else:
        return 0


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def get_ID(nPath):
    '''
    nPath : excel파일 경로

    return : 익명환자ID값만을 반환
    '''
    data = pd.read_excel(nPath, sheet_name='BRC_metastasis_input', dtype=str)
    return data[u'ID']


def get_IDandLR(nPath):
    '''
    nPath : excel파일 경로

    return : 익명환자ID값만을 반환
    '''
    data = pd.read_excel(nPath, sheet_name='BRC_metastasis_input', dtype=str)
    df_filtered = data[[u'ID', u'LR']]
    df_filtered['LR'] = np.where(data['L/R'] == 'L', 1, 0)
    return df_filtered


def get_normalized_metadata(nPath):
    '''
    get original metadata(no normalization)
    nPath : excel파일 경로
    '''
    df_origin = load_excel(nPath)
    df_origin = df_origin.astype({'cT': 'float64'})
    df = df_origin[[u'sex', u'age', u'LR', u'cT', 'cN', u'cAverage', u'cSD', u'aAverage', u'aSD', u'lMax']]

    ## ========= modifying ======== ##
    df['sex'] = np.where(df_origin['sex'] == 'F', 1, 0)
    df['age'] = normalize(df_origin['age'])

    df['LR'] = np.where(df_origin['LR'] == 'L', 1, 0)

    df['homogeneous'] = df_origin.apply(modify_homogeneous, axis=1)
    df['hetero'] = np.where(df_origin['enhancement'] == 'hetero', 1, 0)
    df['rim'] = np.where(df_origin['enhancement'] == 'rim', 1, 0)
    df['clustered'] = np.where(df_origin['enhancement'] == 'clustered', 1, 0)
    df['non-mass'] = np.where(df_origin['enhancement'] == 'non-mass', 1, 0)

    df['cT'] = normalize(df_origin['cT'])
    df['cAverage'] = normalize(df_origin['cAverage'])
    df['cSD'] = normalize(df_origin['cSD'])
    df['aAverage'] = normalize(df_origin['aAverage'])
    df['aSD'] = normalize(df_origin['aSD'])
    df['lMax'] = normalize(df_origin['lMax'])

    df['AorCa'] = normalize(df_origin['aAverage'] - df_origin['cAverage'])
    df['LymAo'] = normalize(df_origin['aAverage'] - df_origin['lMax'])
    df['LymCa'] = normalize(df_origin['cAverage'] - df_origin['lMax'])

    df['cN'] = df_origin.apply(modify_cN, axis=1)

    df['label'] = df_origin[u'pN_modify']

    return df

if __name__ == '__main__':

    print("This is local file")

    # ==== Step 1. Load original dataset
    file_path0 = './BRC_input_210720_train.xlsx'
    file_path1 = './BRC_input_210720_test.xlsx'

    train_excel = load_excel(file_path0)
    print(train_excel)
    #
    #
    #
    # meta_train = sonya.get_normalized_metadata(file_path0)
    # meta_test = sonya.get_normalized_metadata(file_path1)
    # target_dir = './model_06_NN_RFE/'
    # sonya.createFolder(target_dir)



