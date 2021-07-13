#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name : extract_histogram_of_patch.py
@Author : Soyoung Park
@Contect : thegiggles@naver.com
@Time    : 2021-07-07 오후 5:43
@Desc: extracting histogram of patch to measure volume of the enhanced lymph nodes
"""



import os, sys
import numpy as np
import cv2
import argparse
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pydicom as dicom
from scipy import ndimage
import imageio
import pandas as pd
import datetime

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

PATH_PATCH = '/mnt/8TBDisk/BRC/BRC2019_Q_patch'


if __name__=='__main__':
    PATH_PATCH = '/mnt/8TBDisk/BRC/BRC2019_Q_patch'

    print(os.listdir(PATH_PATCH)[0])
    img_path = os.path.join(PATH_PATCH, os.listdir(PATH_PATCH)[0])

    c = Image.open(img_path)
    d = np.array(c)
    print(d.dtype)