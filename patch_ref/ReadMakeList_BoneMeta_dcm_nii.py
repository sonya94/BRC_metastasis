"""
#---------------------------------------------------------------------------#
# File name: ReadMakeList_BoneMeta_dcm_nii.py
#===========================================================================#
#---------------------------*[Description]*---------------------------------#
#  This code enables to make the 3:1 matching list (.pickle) of bone meta
#  data (DCM 3: NII 1).
#---------------------------------------------------------------------------#
"""
import os, glob
import numpy as np
import nibabel
import tensorflow as tf
import pickle

def pause():
    input('Press the <Enter> key to continue...')


def make_or_load_list_of_pickle(Par_data_dir):
    pickle_name = 'Bone_Meta_3to1_matching_list.pickle'
    pickle_file_path = os.path.join(Par_data_dir, pickle_name)

    if not os.path.exists(pickle_file_path):
        print(print('There is no pickle file. \n'))
        print('We are in the data listing step. \n')

        list_of_data = create_data_list(os.path.join(Par_data_dir))

        print("Pickling ...")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(list_of_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Pickle file already exists! I will use this .pickle file \n')

    with open(pickle_file_path, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_data_list(Par_dir):
    if not tf.gfile.Exists(Par_dir):
        print("Could not find the 'input' directory. Set again. \n")
        return None

    sub_par_name_in_Par_dir = ['training', 'validation']
    sub_par_directory_path_list = []
    img_annot_list = {}

    for sub_par_name in sub_par_name_in_Par_dir:
        Par_dir_list = []
        FilePath_list = []
        FileName_list = []

        for ParName, SubName, FileName in os.walk(os.path.join(Par_dir, sub_par_name)):
            if FileName != []:
                Par_dir_list.append(ParName)

                for filename in FileName:
                    FileName_list.append(os.path.join(ParName, filename))

        bundle_record = []
        for folder_name in Par_dir_list:
            nii_file_list = glob.glob(os.path.join(folder_name, '*.gz'))
            if not nii_file_list:
                nii_file_list = glob.glob(os.path.join(folder_name, '*.nii'))

            dcm_file_list = glob.glob(os.path.join(folder_name, '*.dcm'))
            dcm_file_list.sort(reverse=True)


            # nii indexing step
            nii_file = nibabel.load(nii_file_list[0])
            np_array_nii_file = nii_file.get_data()
            no_of_nii_images = np_array_nii_file.shape[2] # index 할 image 수

            for index in range(no_of_nii_images):

                record_list_temp = []

                if np.count_nonzero(np_array_nii_file[:][:, :][:, :, index]) != 0:
                    if index + 1 > no_of_nii_images - 1 or index - 1 < 0:
                        continue
                    record_pre = {'pre_image': dcm_file_list[index - 1], 'pre_annotation': nii_file_list[0],
                                  'pre_annotIndex': index - 1}
                    record_main = {'main_image': dcm_file_list[index], 'main_annotation': nii_file_list[0],
                                'main_annotIndex': index}
                    record_post = {'post_image': dcm_file_list[index + 1], 'post_annotation': nii_file_list[0],
                                   'post_annotIndex': index + 1}

                    record_list_temp.append(record_pre)
                    record_list_temp.append(record_main)
                    record_list_temp.append(record_post)
                    bundle_record.append(record_list_temp)

            img_annot_list[sub_par_name] = bundle_record

    return img_annot_list

if __name__ == '__main__':
    Par_dir = 
    create_data_list()





