sub_par_name_in_Par_dir = ['training', 'validation']
sub_par_directory_path_list = []
img_annot_list = {}
test = 0

Par_dir_list = []
FilePath_list = []
FileName_list = []

## format of directory
## '
## |
## --- case
##      |
##      --- nii
##      --- 0001.dcm
##      --- 0002.dcm

for ParName, SubName, FileName in os.walk(Par_dir):
   if FileName != []:
       Par_dir_list.append(ParName)

       for filename in FileName:
           FileName_list.append(os.path.join(ParName, filename))

bundle_record = []
for folder_name in Par_dir_list:


     nii_file_list = glob.glob(os.path.join(folder_name, '*.gz'))
     print("file_list: {}".format(nii_file_list))
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
             if index + 1 > no_of_nii_images - 1 or index - 1 < 0: # to catch +-1
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
