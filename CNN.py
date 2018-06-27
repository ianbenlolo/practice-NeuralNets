#import dicom_contour.contour as dcm
#import pathlib as Path
import numpy as np
#import matplotlib.patches as patches
#import matplotlib.pyplot as plt

#import pydicom as dicom


import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras import optimizers,regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#from keras_tqdm import TQDMCallback

#import cv2

import logging,warnings
#import glob,os
import data

runName = 'first'
checkpoints_save_filepath="./checkpoints/weights.best.hdf5"


logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='errors.log',level=logging.DEBUG)


# In[86]:


#image_path = './Breast_MRI_cont/Belec/'
#contour_filename = 'RS.1.2.246.352.71.4.139189879485.219269.20180611142708.dcm'


#rt_sequence = dicom.read_file(image_path+contour_filename)
#print rt_sequence


# In[6]:


#dcm.get_roi_names(rt_sequence)
#type(dcm.get_roi_names(rt_sequence))

#y = rt_sequence.ROIContourSequence[1]
#contours = [contour for contour in y.ContourSequence]


#contour_data = dicom.read_file(image_path+contour_filename)

#print dcm.get_roi_names(contour_data)
#img_voxel, mask_voxel = get_data(image_path, contour_filename, roi_index=1)


#image_path = './Breast_MRI_cont/Belec/'

#p=dcm.slice_order(image_path)

#img_voxel, mask_voxel = get_data(image_path, contour_filename, roi_index=1)


#image_path = './Breast_MRI_cont/Belec/'
#contour_filename = 'RS.1.2.246.352.71.4.139189879485.219269.20180611142708.dcm'

#img_voxel, mask_voxel = get_data(image_path, contour_filename, roi_index=1)
#create_dataset(image_shape=512, n_set='train', path='./Breast_MRI_cont'):


#for img, mask in zip(img_voxel, mask_voxel):
#    show_img_msk_fromarray(img, mask, sz=10, cmap='summer_r', alpha=0.7)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def create_model(activ='relu', input_shape = (512,512)):
    """
    Creating basic model with activation. Complex model, goal of overfitting for starters.
    :param activ: Activation function, none if not specified
    :param shape: shape of the model. 512x512 if not specified
    """
    model = models.Sequential()

    model.add(Conv2D(filters = 64,
                     kernel_size = (12,12),
                     activation = activ,
                     strides=(1, 1),
                     padding='valid',
                     input_shape=(input_shape[0], input_shape[1],1)))

    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(filters = 128,
                     kernel_size = (12,12),
                     activation = activ,
                     strides=(1, 1),
                     padding='valid',
                     input_shape=(input_shape[0], input_shape[1],1)))

    model.add(MaxPooling2D(pool_size=(4,4)))

    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([32,32,-1]))
    return model


def training(m, X, Y, verbose=1, batch_size=10, epochs=20):
    """
    Training CNN
    :param m: Keras model
    :param epochs: number of epochs
    :param X: training pictures
    :param Y: training binary ROI mask
    :param verbose: Integer. 0 = silent, 1 = progress bar, 2 = one line per epoch
    :return: history, m
    """
    history = m.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[TQDMCallback()])
    return history

print 'Extracting data...'
img_voxels,img_norm, mask_voxels,roi_squares = data.create_dataset(path = "./Breast_MRI_cont/",roi_name = 'GTV')

print '\n...done\n'
#print (type(roi_squares),roi_squares.shape)
#for img, mask,square in zip(img_voxels, mask_voxels,roi_squares):
#    show_img_msk_fromarray(img, mask,square, sz=10, cmap='summer_r', alpha=0.7)

print 'Creating model'

m = create_model()
gpu_model = keras.utils.multi_gpu_model(m,gpus = 2)

gpu_model.compile(loss='mean_squared_error',
          optimizer='adam',
          metrics=['accuracy'])

#print('Size for each layer :\nLayer, Input Size, Output Size')
#for p in m.layers:
#    print(p.name.title(), p.input_shape, p.output_shape)
print 'Model summary:'
m.summary()


img_norm_arr = np.expand_dims(np.asarray(img_norm), axis=0)
mask_voxels_arr = np.expand_dims(np.asarray(mask_voxels), axis=0)
roi_squares_voxels_arr = np.expand_dims(roi_squares,axis = 0)


img_norm_arr = np.moveaxis(img_norm_arr,0,-1)
mask_voxels_arr = np.moveaxis(mask_voxels_arr,0,-1)
roi_squares_voxels_arr = np.moveaxis(roi_squares_voxels_arr,0,-1)

print ('img array shape:',img_norm_arr.shape)
print ('mask array shape:',mask_voxels_arr.shape)
print ('square mask array shape:',roi_squares_voxels_arr.shape)


#save checkpoints
checkpoint = keras.callbacks.ModelCheckpoint(checkpoints_save_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#to load checkpoints
#model.load_weights("weights.best.hdf5")

print('__FITTING__')
hist = gpu_model.fit(img_norm_arr, roi_squares_voxels_arr, batch_size=10, epochs=40, verbose=2,callbacks=callbacks_list) #callbacks=[TQDMCallback()]

#with open('/home/ianben/trainHistoryDict', 'wb') as file_pi:
#        pickle.dump(hist.history, file_pi)
m.save('/home/ianben/models/'+runName+'.h5')
