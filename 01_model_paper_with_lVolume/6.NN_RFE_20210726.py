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

if __name__ == '__main__':

    print("This is local file")

    # ==== Step 1. Load original dataset
    file_path0 = './BRC_input_210720_train.xlsx'
    file_path1 = './BRC_input_210720_test.xlsx'

    meta_train = sonya.get_normalized_metadata(file_path0)
    meta_test = sonya.get_normalized_metadata(file_path1)
    target_dir = './model_06_NN_RFE/'
    sonya.createFolder(target_dir)

    properties = [  # total 18
        'sex',
        'age',
        'LR',
        'cT',
        'cN',
        'cAverage',
        'cSD',
        'aAverage',
        'aSD',
        'lMax',
        'lVolume',
        'homogeneous',
        'hetero',
        'rim',
        'clustered',
        'non-mass',
        'AorCa',
        'LymAo',
        'LymCa'
    ]

    num_properties = len(properties)

    X_train = meta_train[properties]
    y_train = meta_train['label']

    X_test = meta_test[properties]
    y_test = meta_test['label']

    num_features = len(properties)

    batch_size = 25
    learning_rate = 0.005
    layer1 = 11
    layer2 = 11
    roc_result = 0

    my_model = mlp_model(num_features, lr=learning_rate, l1=layer1, l2=layer2)
    # roc_result = cross_validation(my_model, X_train, y_train, nbatch=batch_size, nlr=learning_rate, l1=layer1, l2=layer2)
    # def cross_validation(model, X, y, nfold=10, nbatch=5, nlr=0.001, l1=16, l2=16):
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    accuracy = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # K-fold cross validation
    # 학습 데이터를 이용해서 학습

    i = 1
    for train_index, validation_index in kfold.split(X_train, y_train):
        kX_train, kX_test = X_train.iloc[train_index], X_train.iloc[validation_index]
        ky_train, ky_test = y_train.iloc[train_index], y_train.iloc[validation_index]

        print("======================batch: {}, lr = {}, FOLD: {}====================".format(batch_size, learning_rate,
                                                                                              i))
        cbks = [callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 ** (epoch // 2)),
                callbacks.TensorBoard(write_graph=False)]
        # hist = model.fit(kX_train, ky_train, epochs=500, batch_size=5, validation_data=(kX_test,ky_test),callbacks=[tb_hist])
        my_model.fit(kX_train, ky_train, epochs=500, batch_size=batch_size, validation_data=(kX_test, ky_test),
                     callbacks=cbks,
                     verbose=0)
        y_val_cat_prob = my_model.predict_proba(kX_test)

        k_accuracy = '%.4f' % (my_model.evaluate(kX_test, ky_test)[1])
        accuracy.append(k_accuracy)

        # roc curve
        fpr, tpr, t = roc_curve(y_train.iloc[validation_index], y_val_cat_prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        i = i + 1

        # 전체 검증 결과 출력

        # Assigning columns namese
        #     cm_df = pd.DataFrame(cm, columns = )

        test_loss, test_acc = my_model.evaluate(kX_test, ky_test)
        #         test_acc_str = 'cross_validation_test acuracy: {}'.format(test_acc)
        # print('Test acuracy: {}'.format(test_acc))
        #         print(test_acc_str)
        print('\nK-fold cross validation Accuracy: {}'.format(test_acc))

    y_pred_proba = (my_model.predict(X_train) >= 0.693).astype(int)
    cm = confusion_matrix(y_train, y_pred_proba)
    print("Radiologist + AI")
    print(cm)
    confusion_metrics(cm)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    roc_result = mean_auc
    print(roc_result)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = target_dir + 'AUC' + str(int(float(mean_auc) * 100)) + "_" + current_time
    model_json = my_model.to_json()

    with open('{}.json'.format(model_name), 'w') as json_file:
        json_file.write(model_json)  # save model per fold

    my_model.save_weights('{}.h5'.format(model_name))  # save weight per fold

    plt.plot(mean_fpr, mean_tpr, color='red', label=r'radiologist + AI(AUC = %0.2f)' % (mean_auc), lw=2, alpha=1)
    plt.show()
    x_bar = np.mean(aucs)
    s = np.std(aucs)
    n = len(aucs)

    z = 1.96
    lower = x_bar - (z * (s / sqrt(n)))
    upper = x_bar + (z * (s / sqrt(n)))
    # interval = z * sqrt( (test_acc * (1 - test_acc)) / len(meta_test))
    print(lower, upper)




