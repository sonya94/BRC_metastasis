#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name : patch_thresholding.py
@Author : Soyoung Park
@Contect : thegiggles@naver.com
@Time    : 2021-07-21 오후 1:31
@Desc: 
"""
def importOwnLib():
    if '/home/miruware/aProjects/lib' not in sys.path:
        sys.path.append('/home/miruware/aProjects/lib')
        print("lib path is successfully appended.")
    else:
        print("lib path is already exists.")

importOwnLib()
import sonyalib as sonya

import importlib
importlib.reload(sonya)
import os, sys
import numpy as np
import pandas as pd
import pydicom as dicom
import cv2
from PIL import Image
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
# %matplotlib inline
# import openpyxl

import datetime
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"


# print(len(result[1]))
def calc_pixels(image):
    total_count = 0
    for i in range(50):
        for j in range(50):
            if image[i][j] > 0:
                total_count += 1

    return total_count


if __name__ == '__main__':

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = str("{}".format(current_time))
    path = '/mnt/8TBDisk/BRC'
    PATH = '/mnt/8TBDisk/BRC/BRC2019_Q_patch'
    PATH_THRES_RESULT = '/mnt/8TBDisk/BRC/BRC2019_Q_thres_result'
    sonya.createFolder(PATH_THRES_RESULT)
    PATH_RECORD = '/mnt/8TBDisk/BRC/lymph_volumns_' + file_name + '.xlsx'
    record_count = 0
    record = pd.DataFrame(columns=["환자key", "lVolume"])
    # print(PATH_RECORD)
    img_list = [list_Q for list_Q in os.listdir(PATH) if list_Q.endswith('.png')]
    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(PATH, img_list[i]), cv2.IMREAD_GRAYSCALE)

        pd.set_option('display.max_rows', img.shape[0] + 1)
        origin = img.copy()
        img2 = img.copy()

        ret, thresh = cv2.threshold(img, 0, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 가장 바깥쪽 컨투어에 대해 모든 좌표 반환
        contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 가장 바깥쪽 컨투어에 대해 꼭지점 좌표만 반환
        contour2, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stencil = np.zeros(img.shape).astype(img.dtype)

        color = [255, 255, 255]
        max_cnt = max(contour2, key=cv2.contourArea)
        cv2.drawContours(img, [max_cnt], 0, (255, 0, 0), thickness=cv2.FILLED)

        cv2.fillPoly(stencil, [max_cnt], color)
        result = cv2.bitwise_and(img, stencil)

        png_name_format = img_list[i].split('.')[0] + '_thres.png'

        plt.imsave(os.path.join(PATH_THRES_RESULT, png_name_format), result)
        print(calc_pixels(result))
        plt.subplot(1, 3, 1)
        plt.imshow(origin, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.imshow(result, cmap='gray')

        new_record = {'환자key': img_list[i].split('_')[0], 'lVolume': str(calc_pixels(result))}
        record.loc[record_count] = new_record
        record_count += 1

        plt.suptitle(img_list[i])
        plt.tight_layout()
        plt.show()

    # record.to_excel(PATH_RECORD, sheet_name='sheet1', index=False)
