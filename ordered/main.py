!git clone https://github.com/seyed-mohammadreza-mousavi/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
%cd Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/ordered/

%run path.py
%run var.py
%run functions.py

print("number of training images:",len(train_image_path_list))
print("number of valid/test images:",len(val_image_path_list))
print("number of testing images:",len(test_image_path_list))

# generate train patch images
for i in tqdm(range(len(train_image_path_list)),desc="Generate the training patches: "):
  image2patch_train(train_image_path_list[i],patch_num,patch_size,training=True,show=False)  # set show=True to visualize the sample process, which is much slower than show=False

for i in tqdm(range(len(val_image_path_list)),desc="Generate the val patches: "):
  image2patch_validation(val_image_path_list[i],patch_num,patch_size,training=False,show=False)

train_patch_img_path_list=sorted(glob(train_patch_dir+"*-*-img.jpg"))
train_patch_groundtruth_path_list=sorted(glob(train_patch_dir+"*-*-groundtruth.jpg"))
train_patch_img_path_list,train_patch_groundtruth_path_list=shuffle(train_patch_img_path_list,train_patch_groundtruth_path_list,random_state=0)

val_patch_img_path_list=sorted(glob(test_patch_dir+"*_*_val_img.jpg"))
val_patch_groundtruth_path_list=sorted(glob(test_patch_dir+"*_*_val_groundtruth.jpg"))

print(len(train_patch_img_path_list),len(train_patch_groundtruth_path_list))
print(train_patch_img_path_list[:2])
print(train_patch_groundtruth_path_list[:2])

print(val_patch_img_path_list[:2])
print(val_patch_groundtruth_path_list[:2])

# Training Dataloader
train_dataset=tf.data.Dataset.from_tensor_slices((train_patch_img_path_list,train_patch_groundtruth_path_list))
train_dataset=train_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

# VAL Dataloader
val_dataset=tf.data.Dataset.from_tensor_slices((val_patch_img_path_list,val_patch_groundtruth_path_list))
val_dataset=val_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset =val_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)




