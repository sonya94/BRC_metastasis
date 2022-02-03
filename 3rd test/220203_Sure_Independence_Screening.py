#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name       : 20210809_Sure_Independence_Screening.py
@Author     : Soyoung Park
@Contect    : thegiggles@naver.com
@Time       : 2021-08-09 오후 7:39
@Version    :
@Desc       : To test whether each feature is reliable to use in model
"""


import warnings
warnings.filterwarnings(action='ignore')
import os, sys
import pandas as pd
import numpy as np

import importlib

def importOwnLib():
    if '/mnt/8TBDisk/github/lib' not in sys.path:
        sys.path.append('/mnt/8TBDisk/github/lib')
        print("lib path is successfully appended.")
    else:
        print("lib path is already exists.")

importOwnLib()
import sonyalib as sonya
importlib.reload(sonya)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import matplotlib.pyplot as plt


def load_excel(nPath):
    return pd.read_excel(nPath, sheet_name='analyze', engine='openpyxl')

def normalize(data):
    return (data-data.min())/(data.max()-data.min())

def get_original_metadata(nPath):
    df_origin = load_excel(nPath)
    properties = [  # total 16
        u'sex', u'age', u'T_site', u'T_size', u'T_homogeneous', u'T_non-homogeneous', u'T_non-mass', u'T_average',
        u'T_SD',
        u'A_average', u'A_SD', u'N_maximum', u'N_volume', u'aorta-tumor', u'aorta-node', u'tumor-node']
    df_origin = df_origin.astype({'T_site': 'float64'}).dropna().reset_index(drop=True)
    df = df_origin[properties]

    df['sex'] = np.where(df_origin['sex'] == 'F', 1, 0)
    df['T_site'] = np.where(df_origin['T_site'] == 'L', 1, 0)

    return df

def get_normalized_metadata(nPath):
    '''
    get normalized metadata
    nPath : excel파일 경로
    '''
    df_origin = load_excel(nPath)
    properties = [  # total 16
        'sex', u'age', 'T_site', u'T_size', u'T_homogeneous', u'T_non-homogeneous', u'T_non-mass', u'T_average', u'T_SD',
        u'A_average', u'A_SD', u'N_maximum', u'N_volume', u'aorta-tumor', u'aorta-node', u'tumor-node']
    # df_origin = df_origin.astype({'T_site': 'float64'}).dropna().reset_index(drop=True)
    df = df_origin[properties]

    ## ========= modifying ======== ##
    df['sex'] = np.where(df_origin['sex'] == 'F', 1, 0)
    df['age'] = normalize(df_origin['age'])

    df['T_site'] = np.where(df_origin['T_site'] == 'L', 1, 0)

    df['T_size'] = normalize(df_origin['T_size'])
    df['T_average'] = normalize(df_origin['T_average'])
    df['T_SD'] = normalize(df_origin['T_SD'])
    df['A_average'] = normalize(df_origin['A_average'])
    df['A_SD'] = normalize(df_origin['A_SD'])
    df['N_maximum'] = normalize(df_origin['N_maximum'])
    df['N_volume'] = normalize(df_origin['N_volume'])

    df['aorta-tumor'] = normalize(df_origin['aorta-tumor'])
    df['aorta-node'] = normalize(df_origin['aorta-node'])
    df['tumor-node'] = normalize(df_origin['tumor-node'])

    df['label'] = df_origin[u'label']

    return df


def createFolder(directory):
    """
    :param directory: directory with name
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

if __name__ == '__main__':
    input_path = '/mnt/8TBDisk/github/BRC_metastasis/3rd test/BRC_input_final_3rd_220126_copy.xlsx'
    input_meta = get_normalized_metadata(input_path)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    properties = [  # total 16
        'sex',
        'age',
        'T_site',
        'T_size',
        'T_homogeneous',
        'T_non-homogeneous',
        'T_non-mass',
        'T_average',
        'T_SD',
        'A_average',
        'A_SD',
        'N_maximum',
        'N_volume',
        'aorta-tumor',
        'aorta-node',
        'tumor-node'
    ]
    num_properties = len(properties)

    X_train_all, X_test_all, y_train, y_test = train_test_split(input_meta[properties], input_meta['label'], test_size=0.33, random_state=44)

    for feature in properties:
        print(feature)
        X_train = X_train_all[feature]
        # y_train = meta_train['label']
        X_train = X_train.values.reshape(-1, 1)

        X_test = X_test_all[feature]
        # y_test = meta_test['label']
        X_test = X_test.values.reshape(-1, 1)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        y_pred_proba = logreg.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)


        plt.plot(fpr, tpr, label="{}, auc={}".format(str(feature) ,str(round(auc, 3))))
        plt.legend(loc=4)

        sonya.createFolder('./SIS_normalized')
        plt.savefig(os.path.join('./SIS_normalized', '%s_%s_%.4f.png' % (time, feature, auc)))
        plt.show()
        # https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python