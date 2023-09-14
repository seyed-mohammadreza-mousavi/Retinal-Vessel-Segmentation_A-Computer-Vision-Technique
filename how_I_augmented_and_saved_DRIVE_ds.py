%cd DRIVE/training/images/
!find . -type f -name '*.tif' ! -name '38_training.tif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
%cd DRIVE/training/1st_manual/
!find . -type f -name '*.gif' ! -name '38_manual1.gif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
%cd DRIVE/training/mask/
!find . -type f -name '*.gif' ! -name '38_training_mask.gif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/

%cd DRIVE/test/images/
!find . -type f -name '*.tif' ! -name '18_test.tif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
%cd DRIVE/test/1st_manual/
!find . -type f -name '*.gif' ! -name '18_manual1.gif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
%cd DRIVE/test/mask/
!find . -type f -name '*.gif' ! -name '18_test_mask.gif' -exec rm -f {} +
%cd /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/

EPOCHS=200
LR=0.0001
BATCH_SIZE=10

num_augmentations_per_image_for_train = 350  # Specify the number of augmentations per image
num_augmentations_per_image_valid = 150

patch_size=48        # patch image size
patch_num=1500        # sample number of one training image
#patch_num=15        # sample number of one training image
patch_threshold=25   # threshold for the patch, the smaller threshoold, the less vessel in the patch
TRAIN_OR_VAL=0.7
dataset_path='DRIVE/'   # modify the dataset_path to your own dir

train_dir=dataset_path+"training/"
test_dir=dataset_path+"test/"

train_image_dir=train_dir+"images/"
train_mask_dir=train_dir+"mask/"
train_groundtruth_dir=train_dir+"1st_manual/"
test_image_dir=test_dir+"images/"
test_mask_dir=test_dir+"mask/"
test_groundtruth_dir=test_dir+"groundtruth/"
test_save_dir=test_dir+"pred_result/"

train_image_path_list=glob(train_image_dir+"*.tif")
test_image_path_list=glob(test_image_dir+"*.tif")

val_image_path_list = test_image_path_list
#val_image_path_list=random.sample(train_image_path_list,int(len(train_image_path_list)*(1-TRAIN_OR_VAL)))
#train_image_path_list=[i for i in train_image_path_list if i not in val_image_path_list]

print("number of training images:",len(train_image_path_list))
print("number of valid/test images:",len(val_image_path_list))
print("number of testing images:",len(test_image_path_list))

def restrict_normalized(imgs,mask):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[2]):
        imgs_normalized[:,:,i] = ((imgs_normalized[:,:,i] - np.min(imgs_normalized[:,:,i])) / (np.max(imgs_normalized[:,:,i])-np.min(imgs_normalized[:,:,i])))*255
    return imgs_normalized
def clahe_equalized(imgs):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  imgs_equalized = np.empty(imgs.shape)
  for i in range(imgs.shape[2]):
    imgs_equalized[:,:,i] = clahe.apply(np.array(imgs[:,:,i], dtype = np.uint8))
  return imgs_equalized
def normalized(imgs):
  imgs_normalized =np.empty(imgs.shape)
  for i in range(imgs.shape[2]):
    imgs_normalized[:,:,i] =cv2.equalizeHist(imgs[:,:,i])
  return imgs_normalized
def adjust_gamma(imgs, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
  new_imgs = np.empty(imgs.shape)
  for i in range(imgs.shape[2]):
    new_imgs[:,:,i] = cv2.LUT(np.array(imgs[:,:,i], dtype = np.uint8), table)
  return new_imgs
def preprocess(image,mask):
  assert np.max(mask)==1
  image=np.array(image)
  image[:,:,0]=image[:,:,0]*mask
  image[:,:,1]=image[:,:,1]*mask
  image[:,:,2]=image[:,:,2]*mask
  image=restrict_normalized(image,mask)
  image=clahe_equalized(image)
  image=adjust_gamma(image,1.2)
  image=image/255.0
  return image
def check_coord(x,y,h,w,patch_size):
  if x-patch_size/2>0 and x+patch_size/2<h and y-patch_size/2>0 and y+patch_size/2<w:
    return True
  return False
def distortion_free_resize(image, img_size=(448, 448)):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=False)
    return image

train_data_path = "DRIVE/training/train_data/"
train_images_preprocessed = []; train_groundtruth = []
for i in tqdm(range(len(train_image_path_list)), desc="preprocessing the training images: "):
  train_image = train_image_path_list[i]
  image_name = train_image.split("/")[-1]
  image_name_number = image_name.split("_")[0]
  image_np = plt.imread(train_image)

  train_mask_path = train_mask_dir+image_name_number+"_training_mask.gif"
  train_mask_path_np = plt.imread(train_mask_path)
  mask=np.where(train_mask_path_np>0,1,0)

  image = preprocess(image_np, mask)
  #image = distortion_free_resize(image, (448, 448))

  groundtruth_image_path = train_dir+"1st_manual/"+image_name_number+"_manual1.gif"
  groundtruth_np = plt.imread(groundtruth_image_path)
  groundtruth=np.where(groundtruth_np>0,255,0)
  #groundtruth=tf.convert_to_tensor((groundtruth/255.0).astype(np.uint8))
  #groundtruth = distortion_free_resize(groundtruth, (448, 448))
  train_images_preprocessed.append(image);train_groundtruth.append(groundtruth)
valid_data_path = "DRIVE/training/valid_data/"
valid_images_preprocessed = []; valid_groundtruth = []
for i in tqdm(range(len(test_image_path_list)), desc="preprocessing the validation/test images: "):
  valid_image = test_image_path_list[i]
  valid_image_name = valid_image.split("/")[-1]
  valid_image_name_number = valid_image_name.split("_")[0]
  valid_image_np = plt.imread(valid_image)

  valid_mask_path = test_mask_dir+valid_image_name_number+"_test_mask.gif"
  valid_mask_path_np = plt.imread(valid_mask_path)
  valid_mask=np.where(valid_mask_path_np>0,1,0)

  valid_image = preprocess(valid_image_np, valid_mask)
  #image = distortion_free_resize(image, (448, 448))

  val_groundtruth_image_path = test_dir+"1st_manual/"+valid_image_name_number+"_manual1.gif"
  val_groundtruth_np = plt.imread(val_groundtruth_image_path)
  val_groundtruth=np.where(val_groundtruth_np>0,255,0)
  #groundtruth=tf.convert_to_tensor((groundtruth/255.0).astype(np.uint8))
  #groundtruth = distortion_free_resize(tf.expand_dims(groundtruth, 2), (448, 448))
  valid_images_preprocessed.append(valid_image);valid_groundtruth.append(val_groundtruth)

def load_image_groundtruth(img_path,groundtruth_path):
  img=tf.io.read_file(img_path)
  img=tf.image.decode_jpeg(img,channels=3)
  img=tf.image.resize(img,[448,48])

  groundtruth=tf.io.read_file(groundtruth_path)
  groundtruth=tf.image.decode_jpeg(groundtruth,channels=1)
  # data argument
  if random.uniform(0,1)>=0.5:
    img=tf.image.flip_left_right(img)
    groundtruth=tf.image.flip_left_right(groundtruth)
#   if random.uniform(0,1)>=0.5:
#     seeds=random.uniform(0,1)
#     img=tf.image.central_crop(img,seeds)
#     groundtruth=tf.image.central_crop(groundtruth,seeds)
  img=tf.image.resize(img,[448,448])
  groundtruth=tf.image.resize(groundtruth,[448,448])
  img/=255.0
  groundtruth=(groundtruth+40)/255.0
  groundtruth=tf.cast(groundtruth,dtype=tf.uint8)
  return img,groundtruth

  num_images = len(train_images_preprocessed)
from tensorflow.keras.layers.experimental import preprocessing
def custom_data_generator(image, groundtruth):
    # Define augmentation layers
    augmentation_layers = keras.Sequential(
        [
            preprocessing.RandomRotation(factor=0.15),  # Rotation within -45 to +45 degrees
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Random translation
            preprocessing.RandomFlip(),  # Random horizontal flip
        ],
        name="img_augmentation",
    )
    image = tf.cast(image, tf.float64)
    groundtruth = tf.cast(tf.expand_dims(groundtruth, 3), tf.float64)
    input_tensor = tf.concat([image, groundtruth], axis=3)

    # Apply the same augmentation to both the image and groundtruth
    augmented_tensor = augmentation_layers(input_tensor)

    augmented_image = augmented_tensor[:, :, :, :3]
    augmented_groundtruth = augmented_tensor[:, :, :, 3:]

    return augmented_image, augmented_groundtruth

t_img_lbl_pair=list(zip(train_images_preprocessed, train_groundtruth))

print(f"Augmenting process takes a little time. Be patient\n")
# Create a list to store augmented images
t_augmented_pairs = []
# Apply augmentations to each image in the dataset
for i, j in tqdm(t_img_lbl_pair, f"Augmenting training_data (creating {num_augmentations_per_image_for_train} samples from every train_data): "):
    t_augmented_pairs.append((i, j))
    for _ in range(num_augmentations_per_image_for_train):
        image = tf.expand_dims(i, axis=0)
        g = tf.expand_dims(j, axis=0)
        augmented_image, augmented_groundtruth = custom_data_generator(image, g)
        #image = tf.expand_dims(i, axis=0)
        #g = tf.expand_dims(j, axis=0)
        #augmented_image=img_augmentation(image)
        #augmented_groundtruth=img_augmentation(g)
        t_augmented_pairs.append((augmented_image[0],augmented_groundtruth[0]))

v_img_lbl_pair=list(zip(valid_images_preprocessed, valid_groundtruth))
v_augmented_pairs = []
# num_augmentations_per_image_valid = 2
# Apply augmentations to each image in the dataset
for i, j in tqdm(v_img_lbl_pair, f"Augmenting validation_data (creating {num_augmentations_per_image_valid} samples from every valid/test data): "):
    v_augmented_pairs.append((i, j))
    for _ in range(num_augmentations_per_image_valid):
        image = tf.expand_dims(i, axis=0)
        g = tf.expand_dims(j, axis=0)
        augmented_image, augmented_groundtruth = custom_data_generator(image, g)
        #image = tf.expand_dims(i, axis=0)
        #g = tf.expand_dims(j, axis=0)
        #augmented_image=img_augmentation(image)
        #augmented_groundtruth=img_augmentation(g)
        v_augmented_pairs.append((augmented_image[0],augmented_groundtruth[0]))

duplicate_train_image_path_list=[path for path in train_image_path_list for _ in range(num_augmentations_per_image_for_train+1)]
duplicate_test_image_path_list=[path for path in test_image_path_list for _ in range(num_augmentations_per_image_valid+1)]

t_img = []
t_g = []
for i, j in t_augmented_pairs:
  t_img.append(i)
  t_g.append(j[:, :])
!rm -rf DRIVE/training/train_data
!mkdir DRIVE/training/train_data
for j in tqdm(range(len(t_img)), "saving augmented train images: "):
  train_image = duplicate_train_image_path_list[j];image_name = train_image.split("/")[-1]
  image_name_number = image_name.split("_")[0]
  if isinstance(t_img[j], np.ndarray):
    plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-img.jpg", t_img[j])
    plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-groundtruth.jpg", t_g[j])
  else:
    t_img=t_img[j].numpy()
    plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-img.jpg", t_img[j])
    t_g=t_g[j].numpy()
    plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-groundtruth.jpg", t_g[j])
v_img = []
v_g = []
for i, j in v_augmented_pairs:
  v_img.append(i)
  v_g.append(j[:, :])
!rm -rf DRIVE/training/valid_data
!mkdir DRIVE/training/valid_data
for j in tqdm(range(len(v_img)), "saving augmented validation/test images: "):
  test_image = duplicate_test_image_path_list[j];image_name = test_image.split("/")[-1]
  image_name_number = image_name.split("_")[0]
  if isinstance(v_img[j], np.ndarray):
    plt.imsave(valid_data_path+image_name_number+"-"+str(j)+"-img.jpg", v_img[j])
    plt.imsave(valid_data_path+image_name_number+"-"+str(j)+"-groundtruth.jpg", v_g[j])
  else:
    v_img=v_img[j].numpy()
    plt.imsave(valid_data_path+image_name_number+"-"+str(j)+"-img.jpg", v_img[j])
    v_g=v_g[j].numpy()
    plt.imsave(valid_data_path+image_name_number+"-"+str(j)+"-groundtruth.jpg", v_g[j])

train_data_img_path_list = sorted(glob(train_data_path+"*-*-img.jpg"))
train_groundtruth_img_path_list = sorted(glob(train_data_path+"*-*-groundtruth.jpg"))
train_data_img_path_list, train_groundtruth_img_path_list = shuffle(train_data_img_path_list, train_groundtruth_img_path_list, random_state=0)

!cp /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/DRIVE/training/train_data/* /content/drive/MyDrive/Colab/ds/train_data/ -R
!cp /content/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/DRIVE/training/valid_data/* /content/drive/MyDrive/Colab/ds/valid_data/ -R
%cd /content/
!rm -rf Retinal-Vessel-Segmentation_A-Computer-Vision-Technique
!git clone https://github.com/seyed-mohammadreza-mousavi/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
%cd Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/