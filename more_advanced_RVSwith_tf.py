!apt-get -qq install git
!git config --global user.email "mohammadreza92299@gmail.com"
!git config --global user.name "Seyed-Mohammadreza-Mousavi"
#!cp drive/MyDrive/Colab/vision_ds/DRIVE ./ -R
!git clone https://github.com/aAmohammadrezaaA/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
%cd Retinal-Vessel-Segmentation_A-Computer-Vision-Technique/
!git remote set-url origin https://aAmohammadrezaaA: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@github.com/aAmohammadrezaaA/Retinal-Vessel-Segmentation_A-Computer-Vision-Technique.git
#!ls
!pip install tqdm
!pip install matplotlib
!pip install opencv-python
!pip install tf-nightly
!pip install scikit-learn
!pip install datetime
!pip install onnxruntime
!pip install -U tf2onnx
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
#auth.authenticate_user()
from oauth2client.client import GoogleCredentials
#creds = GoogleCredentials.get_application_default()
import getpass
#!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
#vcode = getpass.getpass()
#!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
#clearing output in colab

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
from tensorflow.keras.layers import AveragePooling2D,Conv2DTranspose,Input,Add,Conv2D, BatchNormalization,LeakyReLU, Activation, MaxPool2D, Dropout, Flatten, Dense,UpSampling2D,Concatenate,Softmax

%matplotlib inline

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

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
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
  # apply gamma correction using the lookup table
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

def image2patch_train(image_path,patch_num,patch_size,training=True,show=True):
  image_name=image_path.split("/")[-1].split("_")[0]

  image=plt.imread(image_path)

  #groundtruth=plt.imread(train_groundtruth_dir+image_name+"_manual1.gif")
  groundtruth=plt.imread(train_dir+"1st_manual/"+image_name+"_manual1.gif")
  groundtruth=np.where(groundtruth>0,1,0)

  mask=plt.imread(train_mask_dir+image_name+"_training_mask.gif")
  mask=np.where(mask>0,1,0)

  image=preprocess(image,mask)
  #image_binary=0.8*image[:,:,1]+0.2*image[:,:,2]

  image_show=image.copy()
  groundtruth_show=np.zeros_like(image)
  groundtruth_show[:,:,0]=groundtruth.copy()
  groundtruth_show[:,:,1]=groundtruth.copy()
  groundtruth_show[:,:,2]=groundtruth.copy()

  sample_count=0
  sample_index=0

  sample_point=np.where(groundtruth==1)     # generate sample point

  state = np.random.get_state()      # shuffle the coord
  np.random.shuffle(sample_point[0])
  np.random.set_state(state)
  np.random.shuffle(sample_point[1])

  patch_image_list=[]
  patch_groundtruth_list=[]

  while sample_count<patch_num and sample_index<len(sample_point[0]):
    x,y=sample_point[0][sample_index],sample_point[1][sample_index]
    if check_coord(x,y,image.shape[0],image.shape[1],patch_size):
      if np.sum(mask[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2])>patch_threshold:     #select according to the threshold

        patch_image_binary=image[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2,:]   # patch image
        patch_groundtruth=groundtruth[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2]       # patch mask
        #patch_image_binary=np.asarray(0.25*patch_image[:,:,2]+0.75*patch_image[:,:,1])         # B*0.25+G*0.75, which enhance the vessel
        patch_groundtruth=np.where(patch_groundtruth>0,255,0)

        #patch_image_binary =cv2.equalizeHist((patch_image_binary*255.0).astype(np.uint8))/255.0

        patch_image_list.append(patch_image_binary)    # patch image
        patch_groundtruth_list.append(patch_groundtruth)             # patch mask
        if show:
          cv2.rectangle(image_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)  #draw the illustration
          cv2.rectangle(groundtruth_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)
        sample_count+=1

    if show:                                 # visualize the sample process
      plt.figure(figsize=(15,15))
      plt.title("processing: %s"%image_name)
      plt.subplot(121)
      plt.imshow(image_show,cmap=plt.cm.gray)   # processd image
      plt.subplot(122)
      plt.imshow(groundtruth_show,cmap=plt.cm.gray)  #groundtruth of the image, patch is showed as the green square
      plt.show()
      display.clear_output(wait=True)
    sample_index+=1

  for i in range(len(patch_image_list)):
    if training==True:
        plt.imsave(train_patch_dir+image_name+"-"+str(i)+"-img.jpg",patch_image_list[i])
        #print(patch_mask_list[i])
        plt.imsave(train_patch_dir+image_name+"-"+str(i)+"-groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)
    else:
        plt.imsave(train_patch_dir+image_name+"_"+str(i)+"_val_img.jpg",patch_image_list[i])
        #print(patch_mask_list[i])
        plt.imsave(train_patch_dir+image_name+"_"+str(i)+"_val_groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)

# delete original patch images
if not os.path.exists(train_patch_dir):
  os.mkdir(train_patch_dir)
else:
  shutil.rmtree(train_patch_dir)
  os.mkdir(train_patch_dir)

if not os.path.exists(test_save_dir):
  os.mkdir(test_save_dir)

# generate train patch images
for i in tqdm(range(len(train_image_path_list)),desc="Generate the training patches: "):
  image2patch_train(train_image_path_list[i],patch_num,patch_size,training=True,show=False)  # set show=True to visualize the sample process, which is much slower than show=False

def image2patch_validation(image_path,patch_num,patch_size,training=True,show=True):
  image_name=image_path.split("/")[-1].split("_")[0]

  image=plt.imread(image_path)

  #groundtruth=plt.imread(train_groundtruth_dir+image_name+"_manual1.gif")
  groundtruth=plt.imread(test_dir+"1st_manual/"+image_name+"_manual1.gif")
  groundtruth=np.where(groundtruth>0,1,0)

  #mask=plt.imread(train_mask_dir+image_name+"_training_mask.gif")
  mask=plt.imread(test_mask_dir+image_name+"_test_mask.gif")
  mask=np.where(mask>0,1,0)

  image=preprocess(image,mask)
  #image_binary=0.8*image[:,:,1]+0.2*image[:,:,2]

  image_show=image.copy()
  groundtruth_show=np.zeros_like(image)
  groundtruth_show[:,:,0]=groundtruth.copy()
  groundtruth_show[:,:,1]=groundtruth.copy()
  groundtruth_show[:,:,2]=groundtruth.copy()

  sample_count=0
  sample_index=0

  sample_point=np.where(groundtruth==1)     # generate sample point

  state = np.random.get_state()      # shuffle the coord
  np.random.shuffle(sample_point[0])
  np.random.set_state(state)
  np.random.shuffle(sample_point[1])

  patch_image_list=[]
  patch_groundtruth_list=[]

  while sample_count<patch_num and sample_index<len(sample_point[0]):
    x,y=sample_point[0][sample_index],sample_point[1][sample_index]
    if check_coord(x,y,image.shape[0],image.shape[1],patch_size):
      if np.sum(mask[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2])>patch_threshold:     #select according to the threshold

        patch_image_binary=image[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2,:]   # patch image
        patch_groundtruth=groundtruth[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2]       # patch mask
        #patch_image_binary=np.asarray(0.25*patch_image[:,:,2]+0.75*patch_image[:,:,1])         # B*0.25+G*0.75, which enhance the vessel
        patch_groundtruth=np.where(patch_groundtruth>0,255,0)

        #patch_image_binary =cv2.equalizeHist((patch_image_binary*255.0).astype(np.uint8))/255.0

        patch_image_list.append(patch_image_binary)    # patch image
        patch_groundtruth_list.append(patch_groundtruth)             # patch mask
        if show:
          cv2.rectangle(image_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)  #draw the illustration
          cv2.rectangle(groundtruth_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)
        sample_count+=1

    if show:                                 # visualize the sample process
      plt.figure(figsize=(15,15))
      plt.title("processing: %s"%image_name)
      plt.subplot(121)
      plt.imshow(image_show,cmap=plt.cm.gray)   # processd image
      plt.subplot(122)
      plt.imshow(groundtruth_show,cmap=plt.cm.gray)  #groundtruth of the image, patch is showed as the green square
      plt.show()
      display.clear_output(wait=True)
    sample_index+=1

  for i in range(len(patch_image_list)):
    if training==True:
        plt.imsave(test_patch_dir+image_name+"-"+str(i)+"-img.jpg",patch_image_list[i])
        #print(patch_mask_list[i])
        plt.imsave(test_patch_dir+image_name+"-"+str(i)+"-groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)
    else:
        plt.imsave(test_patch_dir+image_name+"_"+str(i)+"_val_img.jpg",patch_image_list[i])
        #print(patch_mask_list[i])
        plt.imsave(test_patch_dir+image_name+"_"+str(i)+"_val_groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)

# delete original patch images
if not os.path.exists(test_patch_dir):
  os.mkdir(test_patch_dir)
else:
  shutil.rmtree(test_patch_dir)
  os.mkdir(test_patch_dir)

if not os.path.exists(test_save_dir):
  os.mkdir(test_save_dir)

# generate validation patch images

for i in tqdm(range(len(val_image_path_list)),desc="Generate the val patches: "):
  image2patch_validation(val_image_path_list[i],patch_num,patch_size,training=False,show=False)


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
    self.conv_init=LinearTransform()
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
    x_linear=self.conv_init(x,training=training)
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
#if not os.path.exists('ckpt'):
#  os.mkdir('ckpt')

#if not os.path.exists(log_path):
#  os.mkdir(log_path)

def load_image_groundtruth(img_path,groundtruth_path):
  img=tf.io.read_file(img_path)
  img=tf.image.decode_jpeg(img,channels=3)
  img=tf.image.resize(img,[patch_size,patch_size])

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

  img=tf.image.resize(img,[patch_size,patch_size])
  groundtruth=tf.image.resize(groundtruth,[patch_size,patch_size])

  img/=255.0
  groundtruth=(groundtruth+40)/255.0
  groundtruth=tf.cast(groundtruth,dtype=tf.uint8)

  return img,groundtruth

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

train_patch_img_path_list=sorted(glob(train_patch_dir+"*-*-img.jpg"))
train_patch_groundtruth_path_list=sorted(glob(train_patch_dir+"*-*-groundtruth.jpg"))
train_patch_img_path_list,train_patch_groundtruth_path_list=shuffle(train_patch_img_path_list,train_patch_groundtruth_path_list,random_state=0)


# make sure that img-list and mask-list is in order
print(len(train_patch_img_path_list),len(train_patch_groundtruth_path_list))
print(train_patch_img_path_list[:2])
print(train_patch_groundtruth_path_list[:2])

val_patch_img_path_list=sorted(glob(test_patch_dir+"*_*_val_img.jpg"))
val_patch_groundtruth_path_list=sorted(glob(test_patch_dir+"*_*_val_groundtruth.jpg"))

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

from IPython.display import clear_output
clear_output()

# check here:
#!rm DRIVE/ckpt/ -rf
#!cp /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/  DRIVE/ckpt/ -R
#ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))
print(f"Training starts from here:\n")
lr_step=0
# check here:
last_val_loss=global_last_val_loss=2e10
#global_last_val_loss=last_val_loss=0.33045151829719543
last_val_acc=last_val_f1=last_val_sp=last_val_se=last_val_prec=last_val_auroc=0

# check here:
best_epoch=e_acc=e_f1=e_sp=e_se=e_prec=e_auroc=-1
#best_epoch=50
# check here:
#for epoch in range(50, EPOCHS):
for epoch in range(EPOCHS):
  start_time_epoch = time.time()
  total_batches_per_epoch =  ((patch_num*20)//BATCH_SIZE)
  total_sam_till_end_of_epoch=((patch_num*20)//BATCH_SIZE)*BATCH_SIZE
  data = [["start of epoch", f"{epoch+1}/{EPOCHS}"], ["batch_size", BATCH_SIZE],
          ["total batches per epoch", total_batches_per_epoch],
		  [f"lowest val_loss at epoch {best_epoch}", last_val_loss],
		  [f"highest val_acc at epoch {e_acc}", last_val_acc],
      [f"highest val_f1 at epoch {e_f1}", last_val_f1],
      [f"highest val_sp at epoch {e_sp}", last_val_sp],
      [f"highest val_se at epoch {e_se}", last_val_se],
      [f"highest val_precision at epoch {e_prec}", last_val_prec],
      [f"highest val_auroc at epoch {e_auroc}", last_val_auroc],
		  ["total samples to see till the end of the epoch", total_sam_till_end_of_epoch], ]
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
  columns = [f"train metrics at end of epoch {epoch+1}", f"train values at epoch {epoch+1}"];myTab = PrettyTable();samples_seen_so_far=tstep*BATCH_SIZE;
  myTab.add_column(columns[0], ["loss", "acc", "f1", "specificity", "sensitivity", "precision", "auroc"])
  myTab.add_column(columns[1], [train_loss.result().numpy(), train_acc.result().numpy(), train_f1.result().numpy(), train_sp.result().numpy(), train_se.result().numpy(), train_precision.result().numpy(), train_auroc.result().numpy()])
  print(myTab)
  for vstep, (patch,groundtruth) in enumerate(val_dataset):
    val_step(lr_step,patch,groundtruth)
    print('\rvalidation results: batch {}, samples seen so far: {} ==> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}, val_sp:{:.4f}, val_se:{:.4f}, val_precision:{:.4f}, val_auroc:{:.4f}'.format(vstep, vstep*BATCH_SIZE, val_loss.result(), val_acc.result(), val_f1.result(), val_sp.result(), val_se.result(), val_precision.result(), val_auroc.result()),end="")
    if val_loss.result()<last_val_loss:
      best_epoch=epoch+1
      !rm -rf DRIVE/ckpt/
      checkpoint_path=os.path.join(checkpoint_dir, f'ep-{epoch+1}_va-{val_loss.result()}')
      ckpt.save(checkpoint_path)
      last_val_loss=val_loss.result().numpy()
    if val_acc.result()>last_val_acc:
      e_acc=epoch+1
      last_val_acc=val_acc.result().numpy()
    if val_f1.result()>last_val_f1:
      e_f1=epoch+1
      last_val_f1=val_f1.result().numpy()
    if val_sp.result()>last_val_sp:
      e_sp=epoch+1
      last_val_sp=val_sp.result().numpy()
    if val_se.result()>last_val_se:
      e_se=epoch+1
      last_val_se=val_se.result().numpy()
    if val_se.result()>last_val_se:
      e_se=epoch+1
      last_val_se=val_se.result().numpy()
    if val_precision.result()>last_val_prec:
      e_prec=epoch+1
      last_val_prec=val_precision.result().numpy()
    if val_auroc.result()>last_val_auroc:
      e_auroc=epoch+1
      last_val_auroc=val_auroc.result().numpy()
  print("\n")
  columns = [f"validation metrics at end of epoch {epoch+1}", f"validation values at epoch {epoch+1}"];myTab = PrettyTable();samples_seen_so_far=tstep*BATCH_SIZE;
  myTab.add_column(columns[0], ["val_loss", "val_acc", "val_f1", "val_specificity", "val_sensitivity", "val_precision", "val_auroc"])
  myTab.add_column(columns[1], [val_loss.result().numpy(), val_acc.result().numpy(), val_f1.result().numpy(), val_sp.result().numpy(), val_se.result().numpy(), val_precision.result().numpy(), val_auroc.result().numpy()])
  print(myTab) 
  if last_val_loss<global_last_val_loss:
    !rm -rf /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/
    !cp DRIVE/ckpt/ /content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/ -R
    print(f"\nvalidation results improved and new checkpoint will be transferred to drive if the path exists.")
    global_last_val_loss=last_val_loss
  else:
    print(f"\nresults did not improve in epoch {epoch+1}. The best results acquired at epoch {best_epoch}.")
    trained_till_epoch=f'/content/drive/MyDrive/Colab/vision_ds/crossentropy_checkpoint/trained_till_epoch_{epoch+1}'
    !touch "$trained_till_epoch"
    #!git add ckpt
    #!git commit -m "checkpoint_to_track"
    #!git push
  end_time_epoch = time.time()
  times = end_time_epoch-start_time_epoch;m, s = divmod(times, 60);h, m = divmod(m, 60)
  print(f"\nThis epoch took ({h}:{m}:{np.round(s)}).\nend of epoch{epoch+1}\n#################################################################################################################")
print(f"\nend of training\n")
  
'''
lr_step=0
last_val_loss=2e10
with log_writer.as_default():
  for epoch in range(EPOCHS):
    # renew the recorder
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    train_f1.reset_states();val_f1.reset_states()
    train_sp.reset_states();val_sp.reset_states()
    train_se.reset_states();val_se.reset_states()
    train_precision.reset_states();val_precision.reset_states()
    train_auroc.reset_states();val_auroc.reset_states()

    # training
    for tstep, (patch,groundtruth) in enumerate(train_dataset):
      train_step(lr_step,patch,groundtruth)

      #tf.summary.scalar("learning_rate", optimizer._decayed_lr(tf.float32).numpy(), step=lr_step)
      tf.summary.scalar("learning_rate", optimizer.lr.numpy(), step=lr_step)
      print('\repoch {}/{:.4f}, batch {}/{:.4f} ==> train_loss:{:.4f}, train_acc:{:.4f}, train_f1:{:.4f}, train_sp:{:.4f}, train_se:{:.4f}, train_precision:{:.4f}, train_auroc:{:.4f}'.format(epoch + 1, EPOCHS, tstep, np.ceil(len(train_patch_img_path_list)/BATCH_SIZE)-1, train_loss.result(), train_acc.result(), train_f1.result(), train_sp.result(), train_se.result(), train_precision.result(), train_auroc.result()),end="")
      lr_step+=1

    if (epoch + 1) % VAL_TIME == 0:
      #valid
      for vstep, (patch,groundtruth) in enumerate(val_dataset):

        val_step(lr_step,patch,groundtruth)

      print('\repoch {}/{:.4f}, batch {}/{:.4f} ==> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}, val_sp:{:.4f}, val_se:{:.4f}, val_precision:{:.4f}, val_auroc:{:.4f}'.format(epoch + 1, EPOCHS, vstep, np.ceil(len(train_patch_img_path_list)/BATCH_SIZE)-1, val_loss.result(), val_acc.result(), val_f1.result(), val_sp.result(), val_se.result(), val_precision.result(), val_auroc.result()),end="")
      tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
      tf.summary.scalar("val_acc", val_acc.result(), step=epoch)

      if val_loss.result()<last_val_loss:
        ckpt.save(checkpoint_path)
        last_val_loss=val_loss.result()
        !cp DRIVE/ckpt drive/MyDrive/Colab/vision_ds/ -Rf
    print("")
    tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
    tf.summary.scalar("train_acc", train_acc.result(), step=epoch)
    log_writer.flush()

# for training from last saved checkpoint
'''
'''
!cp drive/MyDrive/Colab/vision_ds/ckpt DRIVE/ -R
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
start_epoch = optimizer.iterations.numpy() // (len(train_patch_img_path_list)/BATCH_SIZE) + 1
from tensorflow.python.ops.batch_ops import batch
lr_step=0
last_val_loss=2e10
with log_writer.as_default():
  for epoch in range(int(start_epoch), EPOCHS):
    # renew the recorder
    train_loss.reset_states();train_acc.reset_states();val_loss.reset_states()
    val_acc.reset_states();train_f1.reset_states();val_f1.reset_states()
    train_sp.reset_states();val_sp.reset_states();train_se.reset_states()
    val_se.reset_states();train_precision.reset_states();val_precision.reset_states()
    train_auroc.reset_states();val_auroc.reset_states()
    # training
    for tstep, (patch,groundtruth) in enumerate(train_dataset):
      train_step(lr_step,patch,groundtruth)
      #tf.summary.scalar("learning_rate", optimizer._decayed_lr(tf.float32).numpy(), step=lr_step)
      tf.summary.scalar("learning_rate", optimizer.lr.numpy(), step=lr_step)
      print('\repoch {}/{:.4f}, batch {}/{:.4f} ==> train_loss:{:.4f}, train_acc:{:.4f}, train_f1:{:.4f}, train_sp:{:.4f}, train_se:{:.4f}, train_precision:{:.4f}, train_auroc:{:.4f}'.format(epoch + 1, EPOCHS, tstep, np.ceil(len(train_patch_img_path_list)/BATCH_SIZE)-1, train_loss.result(), train_acc.result(), train_f1.result(), train_sp.result(), train_se.result(), train_precision.result(), train_auroc.result()),end="")
      lr_step+=1
    if (epoch + 1) % 2 == 0:
      #valid
      for vstep, (patch,groundtruth) in enumerate(val_dataset):
        val_step(lr_step,patch,groundtruth)
      print('\repoch {}/{:.4f}, batch {}/{:.4f} ==> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}, val_sp:{:.4f}, val_se:{:.4f}, val_precision:{:.4f}, val_auroc:{:.4f}'.format(epoch + 1, EPOCHS, vstep, np.ceil(len(train_patch_img_path_list)/BATCH_SIZE)-1, val_loss.result(), val_acc.result(), val_f1.result(), val_sp.result(), val_se.result(), val_precision.result(), val_auroc.result()),end="")
      tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
      tf.summary.scalar("val_acc", val_acc.result(), step=epoch)
      if val_loss.result()<last_val_loss:
        ckpt.save(checkpoint_path)
        last_val_loss=val_loss.result()
        !cp DRIVE/ckpt drive/MyDrive/Colab/vision_ds/ -Rf
        print(f"\nnew checkpoint saved")
    print("")
    tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
    tf.summary.scalar("train_acc", train_acc.result(), step=epoch)
    log_writer.flush()
'''
