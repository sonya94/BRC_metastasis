#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
Name : extract_dicom_patch.py
Author : Soyoung Park
Contect : thegiggles@naver.com
Time    : 2021-07-07 오후 5:40
Desc:
"""

import os, sys
import pydicom as dicom
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import glob
import shutil
import pandas as pd


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


def get_windowing(data):
    if 'RescaleSlope' in data:
        dicom_fields = [data[('0028', '1050')].value,  # window center
                        data[('0028', '1051')].value,  # window width
                        data[('0028', '1052')].value,  # intercept
                        data[('0028', '1053')].value]  # slope
    else:
        dicom_fields = [data[('0028', '1050')].value,  # window center
                        data[('0028', '1051')].value,  # window width
                        0,  # intercept
                        1]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img[img <= img_min] = img_min
    img[img > img_max] = img_max

    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def dcm2img(dcm, mode=None):
    window_center, window_width, intercept, slope = get_windowing(dcm)
    image_windowed = window_image(dcm.pixel_array, window_center, window_width, intercept, slope)
    if mode == 'RGB' or 'rgb':
        img_gray = Image.fromarray(image_windowed).convert('L')
        img_result = Image.merge("RGB", (img_gray, img_gray, img_gray))
    else:
        img_result = Image.fromarray(image_windowed).convert('L')

    return img_result


def search_ref_dcm(nUID, nPath):
    # nUID : referenced dcm UID of annotation object
    # nPath : path to case which currently working on
    # return: PIL Image (pixel_array from dcm)
    for series_nonQ in [list_series for list_series in os.listdir(nPath) if
                        list_series.find('Q') < 0 and not list_series.startswith(
                                '.')]:  # iterate series except Q series
        for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(PATH_SERIES, series_nonQ)) if
                     list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
            with dicom.read_file(os.path.join(PATH_SERIES, series_nonQ, nDcm)) as dcm:
                if dcm.SOPInstanceUID == matching_uid:  # searching target dcm with Q
                    img = dcm2img(dcm, mode='RGB')
                    return img, series_nonQ, nDcm


def list_files_Q(destpath):
    dcmlist = []
    qlist = []
    for file in os.listdir(destpath):
        if file.find('Q') >= 0:
            qlist.append(file)
        else:
            dcmlist.append(file)

    dcmlist.sort()
    qlist.sort()

    return dcmlist, qlist


if __name__ == '__main__':

    PATH_BASE = '/mnt/8TBDisk/BRC/BRC2019_Q_test'
    label_width = 5
    for nPatient in [list_patient for list_patient in os.listdir(PATH_BASE) if not list_patient.startswith('.')]:  # iterate all patient
        print(nPatient)
        for nCase in [list_case for list_case in os.listdir(os.path.join(PATH_BASE, nPatient)) if not list_case.startswith('.')]:  # search Q file series
            PATH_SERIES = os.path.join(PATH_BASE, nPatient, nCase)
            series_Q = [list_SeriesQ for list_SeriesQ in os.listdir(PATH_SERIES) if list_SeriesQ.find('Q') >= 0][0]
            for nQ in [list_Q for list_Q in os.listdir(os.path.join(PATH_BASE, nPatient, nCase, series_Q)) if list_Q.endswith('.dcm')]:  # iterate all Q(dcm) file
                dcm_q = dicom.read_file(os.path.join(PATH_BASE, nPatient, nCase, series_Q, nQ))
                if "GraphicAnnotationSequence" in dcm_q:
                    for nAnnotation in range(len(dcm_q.GraphicAnnotationSequence)):
                        # ================================= Searching Referenced SOP Instance UID to match dicom file ===============================
                        matching_uid = dcm_q.GraphicAnnotationSequence[nAnnotation].ReferencedImageSequence[0].ReferencedSOPInstanceUID
                        img_ans, name_series, name_dcm = search_ref_dcm(matching_uid, PATH_SERIES)  # matched dcm image per annotation sequence
                        print(name_series)
                        label_ans = ImageDraw.Draw(img_ans)  # set to draw label on image

                        if "GraphicObjectSequence" in dcm_q.GraphicAnnotationSequence[nAnnotation]:
                            for nA_Object in range(
                                    len(dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence)):
                                obj_shape = \
                                dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object][0x711001].value
                                obj_graphic_data = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object].GraphicData
                                print("Object shape is {}".format(obj_shape))

                                if obj_shape == 'RECT':
                                    pt_X = sorted(obj_graphic_data[::2])
                                    pt_Y = sorted(obj_graphic_data[1::2])
                                    ellipse_data = [min(pt_X), min(pt_Y), max(pt_X), max(pt_Y)]
                                    label_ans.rectangle(list(map(int, ellipse_data)), outline="red")

        os.path.join(PATH_BASE, nPatient, nCase, series_Q, nQ)
        draw_img_name = os.path.join(PATH_BASE, nPatient, nCase, series_Q, nQ.split('_')[3].split('.')[0] + "-" + name_series.split('.')[0] + "-" + name_dcm.replace('.dcm', '.png'))
        img_ans.show()
        # cv2.imshow(img_ans)
                        # img_ans.save(draw_img_name)
        break