!apt-get -qq install git
!git config --global user.email "mohammadreza92299@gmail.com"
!git config --global user.name "Seyed-Mohammadreza-Mousavi"
#!cp drive/MyDrive/Colab/vision_ds/DRIVE ./ -R
!git clone https://github.com/seyed-mohammadreza-mousavi/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
%cd Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
!git remote set-url origin https://aAmohammadrezaaA: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@github.com/aAmohammadrezaaA/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
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
VAL_TIME=2
LR=0.0003
BATCH_SIZE=64

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
train_patch_dir=train_dir+"patch/"
test_patch_dir=test_dir+"patch/"

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
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    return image
	
!rm -rf DRIVE/training/train_data
!mkdir DRIVE/training/train_data
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
for j in range(len(train_images_preprocessed)):
  train_image = train_image_path_list[j];image_name = train_image.split("/")[-1]
  image_name_number = image_name.split("_")[0]
  plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-img.jpg", train_images_preprocessed[j])
  plt.imsave(train_data_path+image_name_number+"-"+str(j)+"-groundtruth.jpg", train_groundtruth[j])

!rm -rf DRIVE/training/valid_data
!mkdir DRIVE/training/valid_data
valid_data_path = "DRIVE/training/valid_data/"
valid_images_preprocessed = []; valid_groundtruth = []
for i in tqdm(range(len(test_image_path_list)), desc="preprocessing the training images: "):
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
for j in range(len(valid_images_preprocessed)):
  valid_image = test_image_path_list[j];valid_image_name = valid_image.split("/")[-1]
  valid_image_name_number = valid_image_name.split("_")[0]
  plt.imsave(valid_data_path+valid_image_name_number+"-"+str(j)+"-img.jpg", valid_images_preprocessed[j])
  plt.imsave(valid_data_path+valid_image_name_number+"-"+str(j)+"-groundtruth.jpg", valid_groundtruth[j])
  
  
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
  
train_data_img_path_list = sorted(glob(train_data_path+"*-*-img.jpg"))
train_groundtruth_img_path_list = sorted(glob(train_data_path+"*-*-groundtruth.jpg"))
train_data_img_path_list, train_groundtruth_img_path_list = shuffle(train_data_img_path_list, train_groundtruth_img_path_list, random_state=0)
print(len(train_data_img_path_list)); print(len(train_groundtruth_img_path_list))
print(train_data_img_path_list[:2])
print(train_groundtruth_img_path_list[:2])

train_dataset=tf.data.Dataset.from_tensor_slices((train_data_img_path_list,train_groundtruth_img_path_list))
train_dataset=train_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)
  
valid_data_img_path_list = sorted(glob(valid_data_path+"*-*-img.jpg"))
valid_groundtruth_img_path_list = sorted(glob(valid_data_path+"*-*-groundtruth.jpg"))
print(len(valid_data_img_path_list)); print(len(valid_groundtruth_img_path_list))
print(valid_data_img_path_list[:2])
print(valid_groundtruth_img_path_list[:2])

val_dataset=tf.data.Dataset.from_tensor_slices((valid_data_img_path_list,valid_groundtruth_img_path_list))
val_dataset=val_dataset.map(load_image_groundtruth,num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1300).prefetch(BATCH_SIZE).batch(BATCH_SIZE)

# define the model under eager mode
class LinearTransform(tf.keras.Model):
  def __init__(self, name="LinearTransform"):
    super(LinearTransform, self).__init__(self,name=name)

    self.conv_r=Conv2D(1,kernel_size=3,strides=1,padding='same',use_bias=False)
    self.conv_g=Conv2D(1,kernel_size=3,strides=1,padding='same',use_bias=False)
    self.conv_b=Conv2D(1,kernel_size=3,strides=1,padding='same',use_bias=False)

    self.pool_rc=AveragePooling2D(pool_size=(patch_size,patch_size),strides=1)
    self.pool_gc=AveragePooling2D(pool_size=(patch_size,patch_size),strides=1)
    self.pool_bc=AveragePooling2D(pool_size=(patch_size,patch_size),strides=1)

    self.bn=BatchNormalization()
    self.sigmoid=Activation('sigmoid')
    self.softmax=Activation('softmax')

  def call(self, input,training=True):
    r,g,b=input[:,:,:,0:1],input[:,:,:,1:2],input[:,:,:,2:3]

    rs=self.conv_r(r)
    gs=self.conv_g(g)
    bs=self.conv_r(b)

    rc=tf.reshape(self.pool_rc(rs),[-1,1])
    gc=tf.reshape(self.pool_gc(gs),[-1,1])
    bc=tf.reshape(self.pool_bc(bs),[-1,1])

    merge=Concatenate(axis=-1)([rc,gc,bc])
    merge=tf.expand_dims(merge,axis=1)
    merge=tf.expand_dims(merge,axis=1)
    merge=self.softmax(merge)
    merge=tf.repeat(merge,repeats=48,axis=2)
    merge=tf.repeat(merge,repeats=48,axis=1)

    r=r*(1+self.sigmoid(rs))
    g=g*(1+self.sigmoid(gs))
    b=b*(1+self.sigmoid(bs))

    output=self.bn(merge[:,:,:,0:1]*r+merge[:,:,:,1:2]*g+merge[:,:,:,2:3]*b,training=training)
    return output

class ResBlock(tf.keras.Model):
  def __init__(self,out_ch,residual_path=False,stride=1):
    super(ResBlock,self).__init__(self)
    self.residual_path=residual_path

    self.conv1=Conv2D(out_ch,kernel_size=3,strides=stride,padding='same', use_bias=False,data_format="channels_last")
    self.bn1=BatchNormalization()
    self.relu1=LeakyReLU()#Activation('leaky_relu')

    self.conv2=Conv2D(out_ch,kernel_size=3,strides=1,padding='same', use_bias=False,data_format="channels_last")
    self.bn2=BatchNormalization()

    if residual_path:
      self.conv_shortcut=Conv2D(out_ch,kernel_size=1,strides=stride,padding='same',use_bias=False)
      self.bn_shortcut=BatchNormalization()

    self.relu2=LeakyReLU()#Activation('leaky_relu')

  def call(self,x,training=True):
    xs=self.relu1(self.bn1(self.conv1(x),training=training))
    xs=self.bn2(self.conv2(xs),training=training)

    if self.residual_path:
      x=self.bn_shortcut(self.conv_shortcut(x),training=training)
    #print(x.shape,xs.shape)
    xs=x+xs
    return self.relu2(xs)


class Unet(tf.keras.Model):
  def __init__(self):
    super(Unet,self).__init__(self)
    #self.conv_init=LinearTransform()
    self.conv_init=Conv2D(1,kernel_size=3,strides=1,padding='same',use_bias=False)
    self.resinit=ResBlock(16,residual_path=True)
    self.up_sample=UpSampling2D(size=(2,2),interpolation='bilinear')
    self.resup=ResBlock(32,residual_path=True)

    self.pool1=MaxPool2D(pool_size=(2,2))

    self.resblock_down1=ResBlock(64,residual_path=True)
    self.resblock_down11=ResBlock(64,residual_path=False)
    self.pool2=MaxPool2D(pool_size=(2,2))

    self.resblock_down2=ResBlock(128,residual_path=True)
    self.resblock_down21=ResBlock(128,residual_path=False)
    self.pool3=MaxPool2D(pool_size=(2,2))

    self.resblock_down3=ResBlock(256,residual_path=True)
    self.resblock_down31=ResBlock(256,residual_path=False)
    self.pool4=MaxPool2D(pool_size=(2,2))

    self.resblock=ResBlock(512,residual_path=True)

    self.unpool3=UpSampling2D(size=(2,2),interpolation='bilinear')
    self.resblock_up3=ResBlock(256,residual_path=True)
    self.resblock_up31=ResBlock(256,residual_path=False)

    self.unpool2=UpSampling2D(size=(2,2),interpolation='bilinear')
    self.resblock_up2=ResBlock(128,residual_path=True)
    self.resblock_up21=ResBlock(128,residual_path=False)

    self.unpool1=UpSampling2D(size=(2,2),interpolation='bilinear')
    self.resblock_up1=ResBlock(64,residual_path=True)

    self.unpool_final=UpSampling2D(size=(2,2),interpolation='bilinear')
    self.resblock2=ResBlock(32,residual_path=True)

    self.pool_final=MaxPool2D(pool_size=(2,2))
    self.resfinal=ResBlock(32)

    self.conv_final=Conv2D(1,kernel_size=1,strides=1,padding='same',use_bias=False)
    self.bn_final=BatchNormalization()
    self.act=Activation('sigmoid')

  def call(self,x,training=True):
    #x_linear=self.conv_init(x,training=training)
    x_linear=self.conv_init(x, training=training)
    x=self.resinit(x_linear,training=training)
    x=self.up_sample(x)
    x=self.resup(x,training=training)

    stage1=self.pool1(x)
    stage1=self.resblock_down1(stage1,training=training)
    stage1=self.resblock_down11(stage1,training=training)

    stage2=self.pool2(stage1)
    stage2=self.resblock_down2(stage2,training=training)
    stage2=self.resblock_down21(stage2,training=training)

    stage3=self.pool3(stage2)
    stage3=self.resblock_down3(stage3,training=training)
    stage3=self.resblock_down31(stage3,training=training)

    stage4=self.pool4(stage3)
    stage4=self.resblock(stage4,training=training)

    stage3=Concatenate(axis=3)([stage3,self.unpool3(stage4)])
    stage3=self.resblock_up3(stage3,training=training)
    stage3=self.resblock_up31(stage3,training=training)

    stage2=Concatenate(axis=3)([stage2,self.unpool2(stage3)])
    stage2=self.resblock_up2(stage2,training=training)
    stage2=self.resblock_up21(stage2,training=training)

    stage1=Concatenate(axis=3)([stage1,self.unpool1(stage2)])
    stage1=self.resblock_up1(stage1,training=training)

    x=Concatenate(axis=3)([x,self.unpool_final(stage1)])
    x=self.resblock2(x,training=training)

    x=self.pool_final(x)
    x=self.resfinal(x,training=training)

    seg_result=self.act(self.bn_final(self.conv_final(x),training=training))

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


model=Unet()


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
  

BATCH_SIZE=2
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
  times = end_time_epoch-start_time_epoch;m, s = divmod(times, 60);h, m = divmod(m, 60)
  print(f"\nThis epoch took ({h}:{m}:{np.round(s)}).\nend of epoch{epoch+1}\n#################################################################################################################")
  if (epoch+1)-best_epoch >= early_stopping:
    print(f"\n No improvements in metrics for {early_stopping} epochs. Early stopping")
print(f"\nend of training\n")