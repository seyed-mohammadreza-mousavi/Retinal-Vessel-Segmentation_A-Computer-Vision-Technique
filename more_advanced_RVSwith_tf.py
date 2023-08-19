#from google.colab import drive
#drive.mount('/content/drive')

#!cp drive/MyDrive/Colab/vision_ds/DRIVE ./ -R

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

from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
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

patch_size=48        # patch image size 
patch_num=1000        # sample number of one training image  
patch_threshold=25   # threshold for the patch, the smaller threshoold, the less vessel in the patch 
TRAIN_OR_VAL=0.7
dataset_path='DRIVE/'   # modify the dataset_path to your own dir

train_dir=dataset_path+"training/"
test_dir=dataset_path+"test/"

train_image_dir=train_dir+"images/"
train_mask_dir=train_dir+"mask/"
train_groundtruth_dir=train_dir+"1st_manual/"
train_patch_dir=train_dir+"patch/"

test_image_dir=test_dir+"images/"
test_mask_dir=test_dir+"mask/"
test_groundtruth_dir=test_dir+"groundtruth/"
test_save_dir=test_dir+"pred_result/"

train_image_path_list=glob(train_image_dir+"*.tif")
test_image_path_list=glob(test_image_dir+"*.tif")


val_image_path_list=random.sample(train_image_path_list,int(len(train_image_path_list)*(1-TRAIN_OR_VAL)))
train_image_path_list=[i for i in train_image_path_list if i not in val_image_path_list]

print("number of training images:",len(train_image_path_list))
print("number of valid images:",len(val_image_path_list))
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

def image2patch(image_path,patch_num,patch_size,training=True,show=True):
  image_name=image_path.split("/")[-1].split("_")[0]

  image=plt.imread(image_path)

  groundtruth=plt.imread(train_groundtruth_dir+image_name+"_manual1.gif")
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
    
# generate patch images 
for i in tqdm(range(len(train_image_path_list)),desc="Generate the training patches: "):
  image2patch(train_image_path_list[i],patch_num,patch_size,training=True,show=False)  # set show=True to visualize the sample process, which is much slower than show=False

for i in tqdm(range(len(val_image_path_list)),desc="Generate the val patches: "):
  image2patch(val_image_path_list[i],patch_num,patch_size,training=False,show=False)


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

EPOCHS=200
VAL_TIME=2
LR=0.0003
BATCH_SIZE=64

checkpoint_path=dataset_path+"ckpt/"
log_path=dataset_path+"logs/"

if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)

if not os.path.exists(log_path):
  os.mkdir(log_path)

def load_image_groundtruth(img_path,groundtruth_path):
  img=tf.io.read_file(img_path)
  img=tf.image.decode_jpeg(img,channels=3)
  img=tf.image.resize(img,[patch_size,patch_size])

  groundtruth=tf.io.read_file(groundtruth_path)
  groundtruth=tf.image.decode_jpeg(groundtruth,channels=1)
  
  # data argument (数据增强部分)
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


train_patch_img_path_list=sorted(glob(train_patch_dir+"*-*-img.jpg"))
train_patch_groundtruth_path_list=sorted(glob(train_patch_dir+"*-*-groundtruth.jpg"))
train_patch_img_path_list,train_patch_groundtruth_path_list=shuffle(train_patch_img_path_list,train_patch_groundtruth_path_list,random_state=0)

# make sure that img-list and mask-list is in order 
print(len(train_patch_img_path_list),len(train_patch_groundtruth_path_list))
print(train_patch_img_path_list[:2])
print(train_patch_groundtruth_path_list[:2])

val_patch_img_path_list=sorted(glob(train_patch_dir+"*_*_val_img.jpg"))
val_patch_groundtruth_path_list=sorted(glob(train_patch_dir+"*_*_val_groundtruth.jpg"))

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
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)

# metric record 
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc=tf.keras.metrics.Mean(name='train_acc')
current_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc=tf.keras.metrics.Mean(name='val_acc')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

# checkpoint 
ckpt = tf.train.Checkpoint(model=model)
#ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
#ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

# tensorboard writer （Tensorboard）
log_dir=log_path+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = tf.summary.create_file_writer(log_dir)

def train_step(step,patch,groundtruth):
  with tf.GradientTape() as tape:
        
    linear,pred_seg=model(patch,training=True)
    losses = dice_loss(groundtruth, pred_seg)
    
  # calculate the gradient 
  grads = tape.gradient(losses, model.trainable_variables)
  # bp 
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  # record the training loss and accuracy (loss)
  train_loss.update_state(losses)
  train_acc.update_state(dice(groundtruth, pred_seg))



def val_step(step,patch,groundtruth):
 
  linear,pred_seg=model(patch,training=False)
  losses = dice_loss(groundtruth, pred_seg)
    
  # record the val loss and accuracy (loss)
  val_loss.update_state(losses)
  val_acc.update_state(dice(groundtruth, pred_seg))
  
  tf.summary.image("image",patch,step=step)
  tf.summary.image("image transform",linear,step=step)
  tf.summary.image("groundtruth",groundtruth*255,step=step)
  tf.summary.image("pred",pred_seg,step=step)
  log_writer.flush()

def dice(y_true,y_pred,smooth=1.):
  y_true=tf.cast(y_true,dtype=tf.float32)
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true,y_pred):
  return (1-dice(y_true,y_pred))


lr_step=0
last_val_loss=2e10
with log_writer.as_default():
  for epoch in range(EPOCHS):
    # renew the recorder 
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
  
    # training 
    for tstep, (patch,groundtruth) in enumerate(train_dataset):
      train_step(lr_step,patch,groundtruth)
    
      #tf.summary.scalar("learning_rate", optimizer._decayed_lr(tf.float32).numpy(), step=lr_step)
      tf.summary.scalar("learning_rate", optimizer.lr.numpy(), step=lr_step)
      print('\repoch {}, batch {}, loss:{:.4f}, dice:{:.4f}'.format(epoch + 1, tstep, train_loss.result(), train_acc.result()),end="")
      lr_step+=1

    if (epoch + 1) % VAL_TIME == 0:
      #valid (验证部分)
      for vstep, (patch,groundtruth) in enumerate(val_dataset):
            
        val_step(lr_step,patch,groundtruth)
        
      print('\repoch {}, batch {}, train_loss:{:.4f}, train_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(epoch + 1, vstep, train_loss.result(), train_acc.result(),val_loss.result(), val_acc.result()),end="")
      tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
      tf.summary.scalar("val_acc", val_acc.result(), step=epoch)
    
      if val_loss.result()<last_val_loss:
        ckpt.save(checkpoint_path)
        last_val_loss=val_loss.result()
    print("")
    tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
    tf.summary.scalar("train_acc", train_acc.result(), step=epoch)
    log_writer.flush()
