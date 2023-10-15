from glob import glob
import os

dataset_path='DRIVE/'   # modify the dataset_path to your own dir

train_dir=dataset_path+"training/"

train_image_dir=train_dir+"images/"
train_mask_dir=train_dir+"mask/"
train_groundtruth_dir=train_dir+"1st_manual/"

train_patch_dir=train_dir+"patch/"


test_dir=dataset_path+"test/"

test_image_dir=test_dir+"images/"
test_mask_dir=test_dir+"mask/"
test_groundtruth_dir=test_dir+"groundtruth/"

test_patch_dir=test_dir+"patch/"

test_save_dir=test_dir+"pred_result/"

train_image_path_list=glob(train_image_dir+"*.tif")
test_image_path_list=glob(test_image_dir+"*.tif")

val_image_path_list = test_image_path_list

'''
# delete original patch images
if not os.path.exists(train_patch_dir):
  os.mkdir(train_patch_dir)
else:
  shutil.rmtree(train_patch_dir)
  os.mkdir(train_patch_dir)

if not os.path.exists(test_save_dir):
  os.mkdir(test_save_dir)

if not os.path.exists(test_patch_dir):
  os.mkdir(test_patch_dir)
else:
  shutil.rmtree(test_patch_dir)
  os.mkdir(test_patch_dir)

if not os.path.exists(test_save_dir):
  os.mkdir(test_save_dir)
  '''