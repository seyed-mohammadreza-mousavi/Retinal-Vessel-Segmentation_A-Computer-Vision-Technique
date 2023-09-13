#!apt-get -qq install git
#!git config --global user.email "mohammadreza92299@gmail.com"
#!git config --global user.name "Seyed-Mohammadreza-Mousavi"
#!cp drive/MyDrive/Colab/vision_ds/DRIVE ./ -R
!git clone https://github.com/seyed-mohammadreza-mousavi/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
%cd Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
#!git remote set-url origin https://aAmohammadrezaaA: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@github.com/aAmohammadrezaaA/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
#!ls
!pip install tqdm
!pip install matplotlib
!pip install opencv-python
!pip install scikit-learn
!pip install datetime
from IPython.display import clear_output;clear_output()

from IPython.display import clear_output;clear_output()
from glob import glob
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import tabulate
from tabulate import tabulate
import keras
from keras import layers
import cv2
import os
import datetime
import random
import time
import shutil
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, MaxPooling2D, DepthwiseConv2D, AveragePooling2D,Conv2DTranspose,Input,Add,Conv2D, BatchNormalization,LeakyReLU, Activation, MaxPool2D, Dropout, Flatten, Dense,UpSampling2D,Concatenate,Softmax, Layer
from tensorflow.keras.models import Model

%matplotlib inline
from IPython.display import clear_output;clear_output()

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
def distortion_free_resize(image, img_size):
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
  #groundtruth = distortion_free_resize(groundtruth, (448, 448))
  valid_images_preprocessed.append(valid_image);valid_groundtruth.append(val_groundtruth)

def load_image_groundtruth(img_path,groundtruth_path):
  img=tf.io.read_file(img_path)
  img=tf.image.decode_jpeg(img,channels=3)
  img=tf.image.resize(img,[48,48])

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
  img=tf.image.resize(img,[48,48])
  groundtruth=tf.image.resize(groundtruth,[48,48])
  img/=255.0
  groundtruth=(groundtruth+40)/255.0
  groundtruth=tf.cast(groundtruth,dtype=tf.uint8)
  return img,groundtruth

num_images = len(train_images_preprocessed)

def custom_data_generator(image, groundtruth):
    # Define augmentation layers
    augmentation_layers = keras.Sequential(
        [
            layers.RandomRotation(factor=0.15),  # Rotation within -45 to +45 degrees
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Random translation
            layers.RandomFlip(),  # Random horizontal flip
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
for i, j in tqdm(t_img_lbl_pair, "Augmenting training_data "):
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
for i, j in tqdm(v_img_lbl_pair, "Augmenting validation_data "):
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

# TRAIN Dataloader
train_dataset=tf.data.Dataset.from_tensor_slices((train_data_img_path_list,train_groundtruth_img_path_list))
train_dataset=train_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

print(f"\nnumber of train images: {len(train_data_img_path_list)}")
print(f"number of train masks: {len(train_groundtruth_img_path_list)}")

print(f"\n# make sure that img-list and mask-list for train samples is in order\n")
print(train_data_img_path_list[:2])
print(train_groundtruth_img_path_list[:2])

valid_data_img_path_list = sorted(glob(valid_data_path+"*-*-img.jpg"))
valid_groundtruth_img_path_list = sorted(glob(valid_data_path+"*-*-groundtruth.jpg"))

# VAL Dataloader
val_dataset=tf.data.Dataset.from_tensor_slices((valid_data_img_path_list,valid_groundtruth_img_path_list))
val_dataset=val_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

print(f"\nnumber of validation/test images: {len(valid_data_img_path_list)}")
print(f"number of validation/test masks: {len(valid_groundtruth_img_path_list)}")
print(f"\n# make sure that img-list and mask-list for valid/test samples is in order\n")
print(valid_data_img_path_list[:2])
print(valid_groundtruth_img_path_list[:2])

class custom1(tf.keras.Model):
    def __init__(self):
        super(custom1, self).__init__()

        # Define the encoder layers
        self.encoder_conv_init=Conv2D(1, 3, activation='relu', padding='same')
        #self.encoder_conv_init=LinearTransform()
        #self.encoder_conv1 = ResBlock(32,residual_path=True)
        self.encoder_conv1 = Conv2D(64, 3, activation='relu', padding='same')
        #self.encoder_conv2 = ResBlock(64,residual_path=True)
        self.encoder_conv2 = Conv2D(64, 3, activation='relu', padding='same')
        self.encoder_conv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.encoder_conv4 = Conv2D(32, 3, activation='relu', padding='same')
        self.encoder_conv5 = Conv2D(32, 3, activation='relu', padding='same')
        self.encoder_conv6 = Conv2D(64, 3, (2, 2), activation='relu', padding='same')
        self.encoder_conv7 = Conv2D(64, 3, activation='relu', padding='same')
        #self.encoder_conv8 = ResBlock(64,residual_path=True)
        self.encoder_conv8 = Conv2D(64, 3, activation='relu', padding='same')
        self.encoder_conv9 = Conv2D(128, 3, (2, 2), activation='relu', padding='same')
        self.encoder_conv10 = Conv2D(128, 3, activation='relu', padding='same')
        self.encoder_conv11 = Conv2D(128, 3, activation='relu', padding='same')
        self.encoder_conv12 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.encoder_conv13 = Conv2D(32, 3, activation='relu', padding='same')
        self.encoder_conv14 = Conv2D(32, 3, activation='relu', padding='same')
        self.encoder_conv15 = Conv2D(1, 1, activation='relu', padding='same')   
        self.final = Activation('tanh')

    def call(self, x, training=True):
        # Encoder
        x_linear=self.encoder_conv_init(x, training=training)
        ed = self.encoder_conv1(x_linear, training=training)
        eds1 = self.encoder_conv2(ed, training=training)
        ed = self.encoder_conv3(eds1, training=training)
        ed = self.encoder_conv4(ed, training=training)
        ed = self.encoder_conv5(ed, training=training)
        ed = self.encoder_conv6(ed, training=training)
        ed = Concatenate(axis=3)([ed,eds1])
        ed = self.encoder_conv7(ed, training=training)
        eds2 = self.encoder_conv8(ed, training=training)
        ed = self.encoder_conv9(eds2, training=training)
        ed = self.encoder_conv10(ed, training=training)
        ed = self.encoder_conv11(ed, training=training)
        ed = self.encoder_conv12(ed, training=training)
        ed = Concatenate(axis=3)([ed,eds2])
        ed = self.encoder_conv13(ed, training=training)
        ed = self.encoder_conv14(ed, training=training)
        ed = self.encoder_conv15(ed, training=training)
        seg_result = self.final(ed, training=training)


        return x_linear,seg_result

checkpoint_dir=dataset_path+"ckpt/"
#log_path=dataset_path+"logs/"

if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)

def dice(y_true,y_pred,smooth=1.):
  y_true=tf.cast(y_true,dtype=tf.float32)
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true,y_pred):
  return (1-dice(y_true,y_pred))


def f1_score(y_true, y_pred, smooth=1.):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    precision = intersection / (K.sum(y_pred_f) + smooth)
    recall = intersection / (K.sum(y_true_f) + smooth)
    f1 = (2. * precision * recall) / (precision + recall + smooth)
    return f1

def specificity(y_true, y_pred, smooth=1.):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_negative = K.sum((1 - y_true_f) * (1 - y_pred_f))
    false_positive = K.sum((1 - y_true_f) * y_pred_f)
    spec = true_negative / (true_negative + false_positive + smooth)
    return spec

def sensitivity(y_true, y_pred, smooth=1.):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positive = K.sum(y_true_f * y_pred_f)
    false_negative = K.sum(y_true_f * (1 - y_pred_f))
    se = true_positive / (true_positive + false_negative + smooth)
    return se

def precision(y_true, y_pred, smooth=1.):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    precision = intersection / (K.sum(y_pred_f) + smooth)
    return precision

alpha = 0.25;gamma = 2.0

def focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    focalloss = -tf.reduce_sum(alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return focalloss

model=custom1()

# Learning rate and optimizer
cosine_decay = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=LR, first_decay_steps=12000,t_mul=1000,m_mul=0.5,alpha=1e-5)
optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay)

# loss function
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False) # we won't need this if using other losses

# metric record
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc=tf.keras.metrics.Mean(name='train_acc')
current_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc=tf.keras.metrics.Mean(name='val_acc')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

train_sp = tf.keras.metrics.Mean(name='train_sp');val_sp = tf.keras.metrics.Mean(name='val_sp')
train_f1 = tf.keras.metrics.Mean(name='train_f1');val_f1 = tf.keras.metrics.Mean(name='val_f1')
train_se = tf.keras.metrics.Mean(name='train_se');val_se = tf.keras.metrics.Mean(name='val_se')
train_precision = tf.keras.metrics.Mean(name='train_precision');val_precision = tf.keras.metrics.Mean(name='val_precision')
train_auroc = tf.keras.metrics.Mean(name='train_auroc');val_auroc = tf.keras.metrics.Mean(name='val_auroc')

# checkpoint
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)


# tensorboard writer （Tensorboard）
#log_dir=log_path+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#log_writer = tf.summary.create_file_writer(log_dir)



def train_step(step,patch,groundtruth):
  with tf.GradientTape() as tape:

    linear,pred_seg=model(patch,training=True)
    losses = loss(groundtruth, pred_seg) # crossentropy
    #losses = dice_loss(groundtruth, pred_seg)
    #losses = focal_loss(groundtruth, pred_seg)

  # calculate the gradient
  grads = tape.gradient(losses, model.trainable_variables)
  # bp
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  # record the training loss and accuracy
  train_loss.update_state(losses);train_acc.update_state(dice(groundtruth, pred_seg))
  train_f1.update_state(f1_score(groundtruth, pred_seg));train_sp.update_state(train_sp(groundtruth, pred_seg))
  train_se.update_state(train_se(groundtruth, pred_seg));train_precision.update_state(precision(groundtruth, pred_seg))
  # Calculate AUROC for training
  y_true = np.reshape(groundtruth, (-1))
  y_pred = np.reshape(pred_seg, (-1))
  train_auroc.update_state(y_true, y_pred)


def val_step(step,patch,groundtruth):

  linear,pred_seg=model(patch,training=False)
  losses = loss(groundtruth, pred_seg) # crossentropy
  #losses = dice_loss(groundtruth, pred_seg)
  #losses = focal_loss(groundtruth, pred_seg)

  # record the val loss and accuracy, f1_score
  val_loss.update_state(losses);val_acc.update_state(dice(groundtruth, pred_seg))
  val_f1.update_state(f1_score(groundtruth, pred_seg));val_sp.update_state(val_sp(groundtruth, pred_seg))
  val_se.update_state(val_se(groundtruth, pred_seg));val_precision.update_state(precision(groundtruth, pred_seg))

  # Calculate AUROC for validation
  y_true = np.reshape(groundtruth, (-1))
  y_pred = np.reshape(pred_seg, (-1))
  val_auroc.update_state(y_true, y_pred)

  #tf.summary.image("image",patch,step=step)
  #tf.summary.image("image transform",linear,step=step)
  #tf.summary.image("groundtruth",groundtruth*255,step=step)
  #tf.summary.image("pred",pred_seg,step=step)
  #log_writer.flush()

# check here:
#!rm DRIVE/ckpt/ -rf
#!cp /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/  DRIVE/ckpt/ -R
ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))
print(f"Training starts from here:\n")
early_stopping = 50
lr_step=0
# check here:
last_val_loss=global_last_val_loss=2e10
#global_last_val_loss=last_val_loss=0.4837116777896881
last_val_acc=global_last_val_acc=last_val_f1=global_last_val_f1=last_val_sp=global_last_val_sp=last_val_se=global_last_val_se=last_val_prec=global_last_val_prec=last_val_auroc=global_last_val_auroc=0

# check here:
e_loss=epoch=e_acc=e_f1=e_sp=e_se=e_prec=e_auroc=-1
#e_loss=40;epoch=-1;e_acc=-1;e_f1=-1;e_sp=-1;e_se=-1;e_prec=-1;e_auroc=-1;
best_epoch=0
cumulative_times=0.0
# check here:
#for epoch in range(70, EPOCHS):
for epoch in range(EPOCHS):
  trained_till_epoch=f'/content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/trained_till_epoch_{epoch+1}'
  last_epoch_number = f'/content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/trained_till_epoch_{epoch}'
  start_time_epoch = time.time()
  total_batches_per_epoch =  ((patch_num*20)//BATCH_SIZE)
  total_sam_till_end_of_epoch=((patch_num*20)//BATCH_SIZE)*BATCH_SIZE
  data = [["start of epoch", f"{epoch+1}/{EPOCHS}"], ["batch_size", BATCH_SIZE],
          ["total batches per epoch", total_batches_per_epoch],
		  [f"lowest val_loss occured at epoch {e_loss}", last_val_loss],
		  [f"highest val_acc occured at epoch {e_acc}", last_val_acc],
          [f"highest val_f1 occured at epoch {e_f1}", last_val_f1],
          [f"highest val_sp occured at epoch {e_sp}", last_val_sp],
          [f"highest val_se occured at epoch {e_se}", last_val_se],
          [f"highest val_precision occured at epoch {e_prec}", last_val_prec],
          [f"highest val_auroc occured at epoch {e_auroc}", last_val_auroc],
		  ["total samples to see till the end of the epoch", total_sam_till_end_of_epoch],
          ["best results occured at epoch", best_epoch]]
  col_names = ["#", "start of epoch Info", "values"]
  print(tabulate(data, headers=col_names,tablefmt="fancy_grid"))
  # renew train recorder
  train_loss.reset_states();train_acc.reset_states();train_f1.reset_states()
  train_sp.reset_states();train_se.reset_states();train_precision.reset_states();train_auroc.reset_states()
  # renew validation recorders
  val_sp.reset_states();val_acc.reset_states();val_f1.reset_states();val_se.reset_states()
  val_precision.reset_states();val_auroc.reset_states();val_loss.reset_states()
  # training
  for tstep, (patch,groundtruth) in enumerate(train_dataset):
    train_step(lr_step,patch,groundtruth)
    print('\rtraining results: batch {}, samples seen so far: {}:  ==> train_loss:{:.4f}, train_acc:{:.4f}, train_f1:{:.4f}, train_sp:{:.4f}, train_se:{:.4f}, train_precision:{:.4f}, train_auroc:{:.4f}'.format(tstep, tstep*BATCH_SIZE, train_loss.result(), train_acc.result(), train_f1.result(), train_sp.result(), train_se.result(), train_precision.result(), train_auroc.result()),end="")
  print(f"\n")
  columns = [f"train metrics at end of epoch {epoch+1}", f"train values at end of epoch {epoch+1}"];myTab = PrettyTable();
  myTab.add_column(columns[0], ["loss", "acc", "f1", "specificity", "sensitivity", "precision", "auroc"])
  myTab.add_column(columns[1], [train_loss.result().numpy(), train_acc.result().numpy(), train_f1.result().numpy(), train_sp.result().numpy(), train_se.result().numpy(), train_precision.result().numpy(), train_auroc.result().numpy()])
  print(myTab)
  for vstep, (patch,groundtruth) in enumerate(val_dataset):
    val_step(lr_step,patch,groundtruth)
    print('\rvalidation results: batch {}, samples seen so far: {} ==> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}, val_sp:{:.4f}, val_se:{:.4f}, val_precision:{:.4f}, val_auroc:{:.4f}'.format(vstep, vstep*BATCH_SIZE, val_loss.result(), val_acc.result(), val_f1.result(), val_sp.result(), val_se.result(), val_precision.result(), val_auroc.result()),end="")
    if val_loss.result().numpy()<last_val_loss:
      e_loss=epoch+1
      last_val_loss=val_loss.result().numpy()
    if val_acc.result().numpy()>last_val_acc:
      e_acc=epoch+1
      last_val_acc=val_acc.result().numpy()
    if val_f1.result().numpy()>last_val_f1:
      e_f1=epoch+1
      last_val_f1=val_f1.result().numpy()
    if val_sp.result().numpy()>last_val_sp:
      e_sp=epoch+1
      last_val_sp=val_sp.result().numpy()
    if val_se.result().numpy()>last_val_se:
      e_se=epoch+1
      last_val_se=val_se.result().numpy()
    if val_se.result().numpy()>last_val_se:
      e_se=epoch+1
      last_val_se=val_se.result().numpy()
    if val_precision.result().numpy()>last_val_prec:
      e_prec=epoch+1
      last_val_prec=val_precision.result().numpy()
    if val_auroc.result().numpy()>last_val_auroc:
      e_auroc=epoch+1
      last_val_auroc=val_auroc.result().numpy()
  print("\n")
  columns = [f"validation metrics at end of epoch {epoch+1}", f"validation values at end of epoch {epoch+1}"];myTab = PrettyTable();
  myTab.add_column(columns[0], ["val_loss", "val_acc", "val_f1", "val_specificity", "val_sensitivity", "val_precision", "val_auroc"])
  myTab.add_column(columns[1], [last_val_loss, last_val_acc, last_val_f1, last_val_sp, last_val_se, last_val_prec, last_val_auroc])
  print(myTab)
  if (last_val_loss<global_last_val_loss): 
    global_last_val_loss=last_val_loss
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation loss did not improved. The best validation result occured at epoch {e_loss}.")
  if (last_val_acc>global_last_val_acc): 
    global_last_val_acc=last_val_acc
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation accuracy did not improved. The best validation accuracy occured at epoch {e_acc}.")
  if (last_val_f1>global_last_val_f1): 
    global_last_val_f1=last_val_f1
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation f1 did not improved. The best validation f1 occured at epoch {e_f1}.") 
  if (last_val_sp>global_last_val_sp): 
    global_last_val_sp=last_val_sp
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation sp did not improved. The best validation sp occured at epoch {e_sp}.")
  if (last_val_se>global_last_val_se): 
    global_last_val_se=last_val_se 
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation se did not improved. The best validation sp occured at epoch {e_se}.")
  if (last_val_prec>global_last_val_prec): 
    global_last_val_prec=last_val_prec
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation precision did not improved. The best validation precision occured at epoch {e_prec}.")
  if (last_val_auroc>global_last_val_auroc): 
    global_last_val_auroc=last_val_auroc
    best_epoch=epoch+1
    !rm DRIVE/ckpt/ -rf
    checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}');ckpt.save(checkpoint_path)
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
  else:
    print(f"\nvalidation auroc did not improved. The best validation precision occured at epoch {e_auroc}.")
  !rm -rf "$last_epoch_number"
  !touch "$trained_till_epoch"
  end_time_epoch = time.time()
  times = end_time_epoch-start_time_epoch
  cumulative_times = cumulative_times + times
  m, s = divmod(cumulative_times, 60)
  h, m = divmod(m, 60)
  print(f"\nTill here it took ({h}:{m}:{np.round(s)}).\nend of epoch{epoch+1}\n#################################################################################################################")
  if (epoch+1)-best_epoch >= early_stopping:
    print(f"\n No improvements in metrics for {early_stopping} epochs. Early stopping")
print(f"\nend of training\n")

