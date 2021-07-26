#!/home/miruware/anaconda3/envs/env4_tf1/bin/python
# coding:utf-8

"""
@Name : extracting_center_point.py
@Author : Soyoung Park
@Contect : thegiggles@naver.com
@Time    : 2021-07-09 오후 12:22
@Desc: at widnows (env_dicom)
    Extract center point and patch
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



from skimage import io
# check available callback events
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

refPt = []
thickPt = []
cropping = False

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
    window_image = image_pixels.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def dcm2pixels(dcm):
    hu_image = transform_to_hu(dcm, dcm.pixel_array)
    image_windowed = window_image(dcm, hu_image)
    return image_windowed

def isometric_conversion(medical_image):
    image_windowed = dcm2pixels(medical_image)
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


def search_ref_dcm(nUID, nPath):
    # nUID : referenced dcm UID of annotation object
    # nPath : path to case which currently working on
    # return: PIL Image (pixel_array from dcm)
    for series_nonQ in [list_series for list_series in os.listdir(nPath) if list_series.find('Q') < 0 and not list_series.startswith('.')]:  # iterate series except Q series
        for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(PATH_SERIES, series_nonQ)) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
            with dicom.read_file(os.path.join(PATH_SERIES, series_nonQ, nDcm)) as dcm:
                if dcm.SOPInstanceUID == nUID:  # searching target dcm with Q
                    img = dcm2png(dcm)
                    iso_img = isometric_conversion(dcm)
                    return iso_img, img, series_nonQ, nDcm

# def search_ref_dcm(nUID, nPath):
#     # nUID : referenced dcm UID of annotation object
#     # nPath : path to case which currently working on
#     # return: PIL Image (pixel_array from dcm)
#     for series_nonQ in [list_series for list_series in os.listdir(nPath) if list_series.find('Q') < 0 and not list_series.startswith('.')]:  # iterate series except Q series
#         for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(PATH_SERIES, series_nonQ)) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
#             with dicom.read_file(os.path.join(PATH_SERIES, series_nonQ, nDcm)) as dcm:
#                 if dcm.SOPInstanceUID == nUID:  # searching target dcm with Q
#                     img = dcm2png(dcm)
#                     return img, series_nonQ, nDcm


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


def window_image_old(img, window_center, window_width, intercept, slope):
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


def dcm2png(dcm, mode=None):
    # window_center, window_width, intercept, slope = get_windowing(dcm)
    # image_windowed = window_image_old(dcm.pixel_array, window_center, window_width, intercept, slope)
    hu_image = transform_to_hu(dcm, dcm.pixel_array)
    image_windowed = window_image(dcm, hu_image)
    if mode == 'RGB' or 'rgb':
        img_gray = Image.fromarray(image_windowed).convert('L')
        img_result = Image.merge("RGB", (img_gray, img_gray, img_gray))
    else:
        img_result = Image.fromarray(image_windowed).convert('L')

    return img_result


def search_ref_png(nUID, nPath):
    # nUID : referenced dcm UID of annotation object
    # nPath : path to case which currently working on
    # return: PIL Image (pixel_array from dcm)
    for series_nonQ in [list_series for list_series in os.listdir(nPath) if
                        list_series.find('Q') < 0 and not list_series.startswith('.')]:  # iterate series except Q series
        for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(PATH_SERIES, series_nonQ)) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
            with dicom.read_file(os.path.join(PATH_SERIES, series_nonQ, nDcm)) as dcm:
                if dcm.SOPInstanceUID == matching_uid:  # searching target dcm with Q
                    img = dcm2png(dcm, mode='RGB')
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



def distance(pt1, pt2):  # 두점간의 거리를 구하는 function
    x1, y1 = pt1[0], pt1[1]  # x1, y1 값을 정의
    x2, y2 = pt2[0], pt2[1]  # x2, y2 값을 정의
    return round(math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)))


def click_and_crop(event, x, y, flags, param):
    # refPt와 cropping 변수를 global로 만듭니다.
    global refPt, thickPt, cropping, image, cropped_image

    # 왼쪽 마우스가 클릭되면 (x, y) 좌표 기록을 시작하고
    # cropping = True로 만들어 줍니다.


    # ROI를 그릴 때 필요한 step
    # 0: 중앙좌표 표시
    # 1: 반지금을 입력해 동공부분 표시
    # 2: 홍채영역을 클릭해 원의 두께 입력

    if event == cv2.EVENT_LBUTTONDOWN:  # center position
        print("Center position")
        refPt = [(x, y)]
        # cv2.line(image, refPt[0], refPt[0], (160, 160, 160), 5)
        cv2.imshow("isoImage", image)
        print("refPt : ", refPt)
    elif event == cv2.EVENT_LBUTTONUP:  # 반지름을 입력해 동공부분 표시
        print("show patch")
        cropped_image = image[refPt[0][1] - 25:refPt[0][1] + 25, refPt[0][0] - 25:refPt[0][0] + 25]
        # image = cv2.rectangle(image, (refPt[0][0] - 25, refPt[0][1] - 25), (refPt[0][0] + 25, refPt[0][1] + 25),
        #                       (255, 0, 0))

        cv2.imshow("cropped", cropped_image)
        cv2.imshow("isoImage", image)
    # elif event == cv2.EVENT_MBUTTONUP:  # 원의 두께 입력
    #     print("show patch")
    #     image = cv2.rectangle(image, (refPt[0][0] - 25, refPt[0][0] - 25), (refPt[0][0] + 25, refPt[0][0] + 25), (255,0,0))
    #     cv2.imshow("image", image)


if __name__=='__main__':
    # PATH_ORIGINAL = '/mnt/2TBDisk/WORK/Broadbeam_bluelight/source'
    PATH_RESULT = '/mnt/8TBDisk/BRC/BRC2019_Q_patch'
    sonya.createFolder(PATH_RESULT)
    PATH_FULL_PNG_RESULT = '/mnt/8TBDisk/BRC/BRC2019_Q_full_png'
    sonya.createFolder(PATH_FULL_PNG_RESULT)
    PATH_ORIGINAL = '/mnt/8TBDisk/BRC/BRC2019_Q'
    record = pd.DataFrame(columns=["dcm_name", "center_position"])
    record_count = 0
    if not os.path.exists(PATH_ORIGINAL):  # check the valid of result directory
        print("Original file does not exist\n")

    if not os.path.exists(PATH_RESULT):  # check the valid of result directory
        os.mkdir(PATH_RESULT)  # if not exist, create it

    # =============================================================
    label_width = 5
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
                        img_iso, img_ans, name_series, name_dcm = search_ref_dcm(matching_uid, PATH_SERIES)  # matched dcm image per annotation sequence
                        label_ans = ImageDraw.Draw(img_ans)  # set to draw label on image
                        if "GraphicObjectSequence" in dcm_q.GraphicAnnotationSequence[nAnnotation]:
                            for nA_Object in range(len(dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence)):
                                # obj_shape = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object][0x711001].value
                                obj_shape = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object].NumberOfGraphicPoints
                                obj_graphic_data = dcm_q.GraphicAnnotationSequence[nAnnotation].GraphicObjectSequence[nA_Object].GraphicData

                                # if obj_shape == 'RECT':
                                #     pt_X = sorted(obj_graphic_data[::2])
                                #     pt_Y = sorted(obj_graphic_data[1::2])
                                #     ellipse_data = [min(pt_X), min(pt_Y), max(pt_X), max(pt_Y)]
                                #     label_ans.rectangle(list(map(int, ellipse_data)), outline="red")

                                if obj_shape == 5:  # Rect
                                    # ----- Draw ROI -----
                                    isDrawn = True
                                    polyline_data = []
                                    for i in range(obj_shape):
                                        polyline_data.append(tuple(obj_graphic_data[i * 2:i * 2 + 2]))
                                    label_ans.line(polyline_data, width=label_width, fill='red', joint='curve')



                        if isDrawn == True:
                            draw_img_name = os.path.join(PATH_FULL_PNG_RESULT, name_dcm.split('.')[0] + str('.png'))
                            img_ans.save(draw_img_name, format='PNG')

                            iso_img = Image.fromarray(iso_img)
                            Image.fromarray(img_iso).save('./test_img.tif')
                            print("test")
                            # plt.imshow(img_iso, cmap=plt.cm.bone)
                            # plt.show()

                            # os.path.join(PATH_ORIGINAL, nPatient, nCase, series_Q, nQ)
                            # plt.imshow()
                            # image = np.array(img_iso)
                            image = np.asarray(img_iso)
                            clone = img_iso.copy()
                            cv2.namedWindow("isoImage")
                            cv2.namedWindow("ROI_Image")
                            cv2.setMouseCallback("isoImage", click_and_crop)

                            while True:
                                cv2.imshow("isoImage", image)
                                cv2.imshow("ROI_Image", np.asarray(img_ans))
                                key = cv2.waitKey(1) & 0xFF

                                # 만약 r이 입력되면, 좌표 리셋합니다.
                                if key == ord("r"):
                                    image = clone.copy()
                                # 그린 영역 포함해 이미지를 저장합니다.
                                elif key == ord("s"):
                                    # rename = os.path.join(PATH_RESULT, img_dir)
                                    # cropped_img = image[]
                                    new_record = {'dcm_name' : name_dcm, 'center_position': refPt[0]}
                                    record.loc[record_count] = new_record
                                    record_count += 1
                                    imageio.imwrite(os.path.join(PATH_RESULT, name_dcm.split('.')[0] + str('.png')), cropped_image.astype(np.int16))

                                    # cv2.imwrite(os.path.join(PATH_RESULT, img_dir), image)
                                # 만약 q가 입력되면 작업을 끝냅니다.
                                elif key == ord("q"):
                                    break
                                elif key == ord("x"):
                                    quit()

                            # 모든 window를 종료합니다.q
                            cv2.destroyAllWindows()
    record.to_excel(PATH_RESULT + '/' + str("center_position records") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx', sheet_name='record', index=False)


    # =============================================================
    # img_dir_list = os.listdir(PATH_ORIGINAL)
    #
    # for img_dir in img_dir_list:
    #     if "B." in img_dir:
    #         img_path = os.path.join(PATH_ORIGINAL, img_dir)
    #
    #         # 이미지를 load 합니다.
    #         # image = cv2.imread(args["image"])
    #         image = cv2.imread(img_path)
    #         # 원본 이미지를 clone 하여 복사해 둡니다.
    #         clone = image.copy()
    #         # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    #         cv2.namedWindow("image")
    #         cv2.setMouseCallback("image", click_and_crop)
    #
    #         '''
    #         키보드에서 다음을 입력받아 수행합니다.
    #         - q : 다음 이미지로 넘어갑니다..
    #         - r : 이미지를 초기화 합니다.
    #         - s : 그린 영역을 포함해 이미지를 저장합니다
    #         - x : 프로그램을 종료합니다.
    #         '''
    #         while True:
    #             # 이미지를 출력하고 key 입력을 기다립니다.
    #             cv2.imshow("image", image)
    #             key = cv2.waitKey(1) & 0xFF
    #
    #             # 만약 r이 입력되면, 좌표 리셋합니다.
    #             if key == ord("r"):
    #                 image = clone.copy()
    #             # 그린 영역 포함해 이미지를 저장합니다.
    #             elif key == ord("s"):
    #                 # rename = os.path.join(PATH_RESULT, img_dir)
    #                 cv2.imwrite(os.path.join(PATH_RESULT, img_dir), image)
    #             # 만약 q가 입력되면 작업을 끝냅니다.
    #             elif key == ord("q"):
    #                 break
    #             elif key == ord("x"):
    #                 quit()
    #
    #         # 모든 window를 종료합니다.q
    #         cv2.destroyAllWindows()
