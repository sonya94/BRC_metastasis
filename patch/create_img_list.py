import os 
import pydicom as dicom
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
import glob
import shutil

def search_ref_dcm(nUID, nPath):
    # nUID : referenced dcm UID of annotation object
    # nPath : path to case which currently working on
    # return: PIL Image (pixel_array from dcm)
    for series_nonQ in [list_series for list_series in os.listdir(nPath) if list_series.find('Q') < 0 and not list_series.startswith('.')]:  # iterate series except Q series
        for nDcm in [list_dcm for list_dcm in os.listdir(os.path.join(nPath, series_nonQ)) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]:  # check extension
            with dicom.read_file(os.path.join(nPath, series_nonQ, nDcm)) as dcm:
                if dcm.SOPInstanceUID == nUID:  # searching target dcm with Q
                    img = dcm2img(dcm, mode='RGB')
                    return img, series_nonQ, nDcm


def create_data_list(path): 
    img_target_list = []
    for root, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        files.sort()
        if dirs == []:
            if root.split(os.path.sep)[-1].find('Q') >= 0: # get annotation data from Q(presentation) file
                series_Q = [list_SeriesQ for list_SeriesQ in sorted(files) if list_SeriesQ.endswith('.dcm') > 0][0]
                q_file = os.path.join(os.path.abspath(PATH_BASE), root, series_Q)
               # print(os.path.join(os.path.abspath(PATH_BASE), root))
                with dicom.read_file(q_file) as dcm_Q:
                    if "GraphicAnnotationSequence" in dcm_Q:
                        dAnnotation = dcm_Q.GraphicAnnotationSequence # save graphic annotations data
                    else:
                        print("ERROR: {}".format(root))
            else: # search target dcm file matched from Q(presentation) file
                record_list_temp = []
                for nAnnotation in range(len(dAnnotation)): # each annotation
                    UID_annot = dAnnotation[nAnnotation].ReferencedImageSequence[0].ReferencedSOPInstanceUID  # UID of target dcm
                    list_dcm = [list_dcm for list_dcm in sorted(files) if list_dcm.endswith('.dcm') and not list_dcm.startswith('.')]
                    for index, nDcm in enumerate(list_dcm):  # check extension
                        with dicom.read_file(os.path.join(root, nDcm)) as dcm:
                            if dcm.SOPInstanceUID == UID_annot:  # searching target dcm with Q
                                record = {'Case': root.split(os.path.sep)[4], 'nAnnotation' : nAnnotation ,'pre' : list_dcm[index-1], 'main' : list_dcm[index], 'post' : list_dcm[index + 1]}
                                record_list_temp.append(record)
                img_target_list.append(record_list_temp)
                           
    return img_target_list

# if __name__ == '__main__':
#     annotation_list = create_data_list(PATH_BASE)
