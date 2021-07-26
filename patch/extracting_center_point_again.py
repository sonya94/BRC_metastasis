#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name : extracting_center_point_again.py
@Author : Soyoung Park
@Contect : thegiggles@naver.com
@Time    : 2021-07-13 오후 1:00
@Desc: re-extract patch based on center position
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
import glob
from pydicom.pixel_data_handlers.util import apply_voi_lut

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

def transform_to_hu(medical_image, image_pixels):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image_pixels * slope + intercept
    return hu_image

def window_image(medical_image, image_pixels):
    window_center = medical_image.WindowCenter
    window_width = medical_image.WindowWidth
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = image_pixels.copy()
    windowed_image[windowed_image < img_min] = img_min
    windowed_image[windowed_image > img_max] = img_max
    return windowed_image

def dcm_rescaling(dcm): # hu_transform + windowing
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    hu_image = dcm.pixel_array * slope + intercept

    window_center = dcm.WindowCenter
    window_width = dcm.WindowWidth
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = hu_image.copy()
    windowed_image[windowed_image < img_min] = img_min
    windowed_image[windowed_image > img_max] = img_max

    return windowed_image

def dcm2pixels(dcm):
    hu_image = transform_to_hu(dcm, dcm.pixel_array)
    # image_windowed = apply_voi_lut(hu_image, dcm)
    image_windowed = window_image(dcm, hu_image)
    return image_windowed

def isometric_conversion(medical_image):
    image_windowed = dcm_rescaling(medical_image)
    # ----- iso pixel spacing -----
    origin_spacing = medical_image.PixelSpacing
    target_spacing = 0.2
    new_spacing = [target_spacing, target_spacing]  # set target spacing value
    # -----------------Resampling----------------- #
    origin_spacing = np.array([float(origin_spacing[0]), float(origin_spacing[1])])  # modify the format of spacing
    resize_factor = origin_spacing / new_spacing
    new_real_shape = image_windowed.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image_windowed.shape
    new_spacing = origin_spacing / real_resize_factor

    iso_image = ndimage.interpolation.zoom(image_windowed, real_resize_factor)  # modified pixel array
    return iso_image


def search_ref_dcm(nUID, nPath): ## for re-extracting patch
    # nUID : referenced dcm UID of annotation object
    # nPath : path to case which currently working on
    # return: PIL Image (pixel_array from dcm)
    for series_nonQ in [list_series for list_series in os.listdir(nPath) if list_series.find('Q') < 0 and not list_series.startswith('.')]:  # iterate series except Q series
        for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(PATH_SERIES, series_nonQ)) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
            with dicom.read_file(os.path.join(PATH_SERIES, series_nonQ, nDcm)) as dcm:
                if dcm.SOPInstanceUID == nUID:  # searching target dcm with Q
                    iso_img = isometric_conversion(dcm)
                    return iso_img, series_nonQ, nDcm


if __name__=='__main__':
    PATH_ORIGINAL = '/mnt/8TBDisk/BRC/BRC2019_Q'
    PATH_RESULT = '/mnt/8TBDisk/BRC/BRC2019_Q_patch_2'
    sonya.createFolder(PATH_RESULT)
    PATH_FULL_PNG_RESULT = '/mnt/8TBDisk/BRC/BRC2019_Q_full_png'
    sonya.createFolder(PATH_FULL_PNG_RESULT)
    PATH_EXCEL = '/mnt/8TBDisk/BRC/center_position_records_20210713.xlsx'
    list_center = pd.read_excel(PATH_EXCEL, sheetname='record', dtype={u'dcm_name': str, u'x': int, u'y': int})
    i = 0

    for nPatient in [list_patient for list_patient in os.listdir(PATH_ORIGINAL) if not list_patient.startswith('.')]:  # iterate all patient
        print(nPatient)
        for nCase in [list_case for list_case in os.listdir(os.path.join(PATH_ORIGINAL, nPatient)) if not list_case.startswith('.')]:  # search Q file series
            PATH_SERIES = os.path.join(PATH_ORIGINAL, nPatient, nCase)
            series_Q = [list_SeriesQ for list_SeriesQ in os.listdir(PATH_SERIES) if list_SeriesQ.find('Q') >= 0][0]
            pixel_spacing = 0
            for nQ in [list_Q for list_Q in os.listdir(os.path.join(PATH_ORIGINAL, nPatient, nCase, series_Q)) if list_Q.endswith('.dcm')]:  # iterate all Q(dcm) file
                dcm_q = dicom.read_file(os.path.join(PATH_ORIGINAL, nPatient, nCase, series_Q, nQ))
                if "GraphicAnnotationSequence" in dcm_q:
                    for nAnnotation in range(len(dcm_q.GraphicAnnotationSequence)):
                        isDrawn = False
                        # ================================= Searching Referenced SOP Instance UID to match dicom file ===============================
                        matching_uid = dcm_q.GraphicAnnotationSequence[nAnnotation].ReferencedImageSequence[0].ReferencedSOPInstanceUID
                        img_iso, name_series, name_dcm = search_ref_dcm(matching_uid, PATH_SERIES)  # matched dcm image per annotation sequence

                        if "GraphicObjectSequence" in dcm_q.GraphicAnnotationSequence[nAnnotation]:
                            for nA_Object in range(len(dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence)):
                                # obj_shape = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object][0x711001].value
                                obj_shape = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object].NumberOfGraphicPoints
                                obj_graphic_data = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object].GraphicData

                                if obj_shape == 5:  # Rect
                                    # ----- Draw ROI -----
                                    isDrawn = True

                        index_coordinates = list_center.index[list_center['dcm_name'] == name_dcm].tolist()
                        if isDrawn and len(index_coordinates) > 0:
                            df_coordinates = list_center.loc[int(index_coordinates[0])]

                            # print(df_coordinates)
                            refPt = (df_coordinates['x'], df_coordinates['y'])
                            cropped_image = img_iso[refPt[1] - 25:refPt[1] + 25, refPt[0] - 25:refPt[0] + 25]
                            print(img_iso[refPt[1] - 25][refPt[0] - 25])
                            imageio.imwrite(os.path.join(PATH_RESULT, name_dcm.split('.')[0] + str('.png')), cropped_image.astype(np.int32))
                            print("image cropped {}".format(i))

                            i += 1
                            #
                            # draw_img_name = os.path.join(PATH_FULL_PNG_RESULT, name_dcm.split('.')[0] + str('.png'))
                            # img_ans.save(draw_img_name, format='PNG')


