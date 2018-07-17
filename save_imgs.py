import numpy as np
"""
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
"""
import sys,os
import data
import cv2


path = '/Users/ianbenlolo/Documents/Hospital/'
path_to_imgs = path+'Breast_MRI_test/'
save_path = path + 'Breast_MRI_cont_png/'
print('hello')

if 'Breast_MRI_cont_png' not in os.listdir(path):
	os.mkdir(path+'Breast_MRI_cont_png')

for file in data.listdir_nohidden(path_to_imgs):
	path_ = file
	print('Starting file: ',file)
	sys.stdout.flush()
	images, images_norm, contour_mask,roi_squares, filenames = data.create_dataset(path=path_, roi_name = 'GTV', ret_files = True)
	print('Extracted.')
	sys.stdout.flush()

	print('images_norm shape: ',np.asarray(images_norm).shape, ' type: ', type(images_norm))
	print('roi_squares shape: ',np.asarray(roi_squares).shape, ' type: ', type(roi_squares))
	print('filenames length: ',np.asarray(filenames).shape,' type: ', type(filenames))
	sys.stdout.flush()

	#inner_file_lengths = [ [len(i) for i in data_ ] for data_ in [data.listdir_nohidden(path_to_imgs+file+ j) for j in filenames]]
	#print('inner_file_lengths shape: ',np.asarray(inner_file_lengths).shape, ' type: ',type(inner_file_lengths), ' ', inner_file_lengths)
	
	assert(len(images_norm) == len(roi_squares) == len(contour_mask)), "werent the same"

	"""
	bins=[]
	for i in range(0,len(inner_file_lengths)):
		if i ==0:
			inner = [(img,contour,rois) for (img,contour,rois) in (images_norm[0:inner_file_lengths[i]],contour_mask[0:inner_file_lengths[i]] roi_squares[0:inner_file_lengths[i]])]
		else:
			inner = [(img,contour,rois) for (img,contour,rois) in (images_norm[inner_file_lengths[i-1]:inner_file_lengths[i]],contour_mask[inner_file_lengths[i-1]:inner_file_lengths[i]] roi_squares[inner_file_lengths[i-1]:inner_file_lengths[i]])]
	"""
	i=0
	print('Saving..')
	for (img,contour,roi) in zip(images_norm, contour_mask,roi_squares):
		img_save = save_path+file + '.'+ str(i) + '_img.png'
		roi_save = save_path+file + '.' + str(i)+ '_roi.png'
		cont_save = save_path+file + '.' + str(i) + '_contour.png'

		cv2.imwrite(img_save, np.asarray(img))
		cv2.imwrite(roi_save, np.asarray(roi))
		cv2.imwrite(cont_save, np.asarray(contour))
		print('Saved. ',i)
		sys.stdout.flush()
		i+=1
	print('Done.')
"""
images, images_norm, contour_mask,roi_squares = data.create_dataset( path='./bla/', roi_name = 'GTV', ret_files = False)
i = 0
print('images_norm shape: ',np.asarray(images_norm).shape, ' type: ', type(images_norm))
print('roi_squares shape: ',np.asarray(roi_squares).shape, ' type: ', type(roi_squares))
print('contour_mask length: ',np.asarray(contour_mask).shape,' type: ', type(contour_mask))
for img, mask,square in zip(images_norm, contour_mask,roi_squares):
	data.show_img_msk_fromarray(img, mask,square, sz=10, cmap='summer_r', alpha=0.7,save_path = '/Users/ianbenlolo/Documents/Hospital/images/'+str(i)+'.png')
	i+=1
"""
def create_model(activ='relu', input_shape = (512,512)):
    """
    Creating basic model with activation. Complex model, goal of overfitting for starters.
    :param activ: Activation function, none if not specified
    :param shape: shape of the model. 512x512 if not specified
    """
    model = models.Sequential()

    model.add(Conv2D(filters = 32,
                     kernel_size = (12,12),
                     activation = activ,
                     strides=(1, 1),
                     padding='valid',
                     input_shape=(input_shape[0], input_shape[1],1)))

    model.add(MaxPooling2D(pool_size=(2,2)))
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

    model.add(Conv2D(filters = 256,
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
