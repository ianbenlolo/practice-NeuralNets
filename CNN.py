#import dicom_contour.contour as dcm
#import pathlib as Path
import numpy as np
#import matplotlib.patches as patches
#import matplotlib.pyplot as plt

#import pydicom as dicom
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras import optimizers,regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import sys
import pickle


#from keras_tqdm import TQDMCallback

#import cv2
import logging,warnings
#import glob,os
import data 

runName = 'empty_dataset_long'
batch_size = 16
epochs = 220

print 'runname: ',runName,' Batch size: ',batch_size, ' epochs: ' , epochs 

if len(sys.argv)>1: #on home computer
	dir_imgs = '/Users/ianbenlolo/Documents/Hospital/Breast_MRI.../'
	checkpoints_save_filepath = '/Users/ianbenlolo/Documents/Hospital/checkpoints/'
else:
    checkpoints_save_filepath="./checkpoints/weights.best.hdf5"
    path_to_contours = "/home/ianben/Breast_MRI_cont_dcm/"
    dir_imgs = '/home/ianben/Breast_MRI_cont_png/images/'
    dir_rois = '/home/ianben/Breast_MRI_cont_png/rois/'


logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='errors.log',level=logging.DEBUG)


print('\nConfigurating session')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

print('Done config protocol\n')


sys.stdout.flush()


def create_model(activ='relu', input_shape = (512,512)):
	model = models.Sequential()
	model.add(Conv2D(filters = 8,
		kernel_size = (12,12),
		activation = activ,
		strides=(1, 1),
		padding='valid',
		input_shape=(input_shape[0], input_shape[1],1)))

	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters = 16,
	kernel_size = (12,12),
	activation = activ,
	strides=(1, 1),
	padding='valid',
	input_shape=(input_shape[0], input_shape[1],1)))

	model.add(MaxPooling2D(pool_size=(6,6)))
	# model.add(Conv2D(filters = 128,
	# 	kernel_size = (12,12),
	# 	activation = activ,
	# 	strides=(1, 1),
	# 	padding='valid',
	# 	input_shape=(input_shape[0], input_shape[1],1)))
	# model.add(MaxPooling2D(pool_size=(4,4)))
	model.add(Flatten())
	model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
	model.add(Reshape([32,32,1]))
	return model


def training(m, X, Y, callbacks,verbose=2, batch_size=10, epochs=20):
    """
    Training CNN
    :param m: Keras model
    :param epochs: number of epochs
    :param X: training pictures
    :param Y: training binary ROI mask
    :param verbose: Integer. 0 = silent, 1 = progress bar, 2 = one line per epoch
    :return: history, m
    """
    history = m.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)
    return history


img_voxel,_, _ ,roi_squares = data.create_empty_dataset(path = path_to_contours,roi_name = 'GTV')
# img_voxel,_, _ ,roi_squares = data.get_data('/home/ianben/Breast_MRI_cont_dcm/032/',data.get_contour_file('/home/ianben/Breast_MRI_cont_dcm/032/'), roi_name = 'GTV')

# #####
# plt.imsave('./hello.png',img_voxel[24])
# plt.imsave('hello1.png',roi_squares[24])
# #####

#mask_voxels_arr = np.expand_dims(np.asarray(mask_voxels), axis=3)

img_voxel_arr = np.asarray(np.expand_dims(np.asarray(img_voxel), axis=3))
roi_squares_voxels_arr = np.asarray(np.expand_dims(roi_squares,axis = 3))

# #####
# img_voxel_arr = np.tile(img_voxel_arr[24],(1000,1,1,1))
# roi_squares_voxels_arr = np.tile(roi_squares_voxels_arr[24],(1000,1,1,1))
# #####

print ('img array shape:',img_voxel_arr.shape)
#print ('mask array shape:',mask_voxels_arr.shape)
print ('square mask array shape:',roi_squares_voxels_arr.shape)



sys.stdout.flush()


data_gen_args = dict(featurewise_center=False,
				 featurewise_std_normalization=False,
				 rotation_range=90.,
				 #width_shift_range=0.1,
				 #height_shift_range=0.1,
				 zoom_range=0.2
				 )

image_datagen = ImageDataGenerator(**data_gen_args)
roi_datagen = ImageDataGenerator(**data_gen_args)


# image_datagen.fit(images, augment=True, seed=seed)
# roi_datagen.fit(masks, augment=True, seed=seed)

# image_generator = image_datagen.flow_from_directory(
#                     dir_imgs, 
#                     target_size=(512,512), 
#                     #color_mode='grayscale',
#                     classes=None, 
#                     class_mode='categorical', 
#                     batch_size=10, 
#                     shuffle=True, 
#                     seed=None, 
#                     save_to_dir=None, 
#                     save_prefix='', 
#                     save_format='png', 
#                     follow_links=False, 
#                     subset=None, 
#                     interpolation='nearest')

# rois_generator = roi_datagen.flow_from_directory(
#                     dir_rois, 
#                     target_size=(32,32), 
#                     #color_mode='grayscale',
#                     classes=None, 
#                     class_mode='categorical', 
#                     batch_size=10, 
#                     shuffle=True, 
#                     seed=None, 
#                     save_to_dir=None, 
#                     save_prefix='', 
#                     save_format='png', 
#                     follow_links=False, 
#                     subset=None, 
#                     interpolation='nearest')



#print (type(roi_squares),roi_squares.shape)
# for img, mask,square in zip(img_voxels, mask_voxels,roi_squares):
#    data.show_img_msk_fromarray(img, mask,square, sz=10, cmap='summer_r', alpha=0.7,save_path = '/Users/ianbenlolo/Documents/Hospital/images/')



print 'Creating model'
m = create_model()


# if len(sys.argv)>1:
#     m.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])


print('Size for each layer :\nLayer, Input Size, Output Size')
for p in m.layers:
   print(p.name.title(), p.input_shape, p.output_shape)

print 'Model summary:'
m.summary()

sys.stdout.flush()


#save checkpoints
checkpoint = keras.callbacks.ModelCheckpoint(checkpoints_save_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#to load checkpoints
#model.load_weights("weights.best.hdf5")

print('__FITTING__')
sys.stdout.flush()

if len(sys.argv)==1:
    gpu_model = keras.utils.multi_gpu_model(m,gpus = 2)

    gpu_model.compile(loss='mean_squared_error',optimizer='SGD',metrics=['accuracy'])

    history = training(gpu_model,img_voxel_arr,roi_squares_voxels_arr,callbacks = callbacks_list, batch_size=batch_size, epochs= epochs)

    # history = gpu_model.fit_generator(
    #     zip(image_generator, rois_generator),
    #     steps_per_epoch=20,
    #     epochs=200)

    # with open('/home/ianben/models/'+runName+'_hist', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

elif len(sys.argv)>1:
    m.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # history = m.fit_generator(
    #     zip(image_generator, rois_generator),
    #     steps_per_epoch=20,
    #     epochs=200)
    
    history = training(m, img_voxel_arr,roi_squares_voxels_arr,callbacks = callbacks_list, batch_size=batch_size, epochs= epochs)

    with open('~/Users/ianbenlolo/Documents/models/'+runname+'_hist', 'wb') as file_pi:
	   pickle.dump(history.history, file_pi)

#callbacks=[TQDMCallback()]

m.save('/home/ianben/models/'+runName+'.h5')

print 'Done.'

