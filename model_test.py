import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np

import keras
from keras.models import Model

import data,sys

img_save_path =  '/home/ianben/images/test_5/'# /Users/ianbenlolo/Documents/Hospital/images/'
model_filename = 'empty_dataset.h5'
img_names = 'empty_set_'
path_to_contours = "/home/ianben/Breast_MRI_cont_dcm_test/"


def load_model(filename):
	print ('Loading model...')
	model = keras.models.load_model('/home/ianben/models/'+ filename)
	print ('Model successfully loaded.')
	return model

def load_data(data_path):
	print ('loading dataset...')
	img_voxels,img_norm, mask_voxels,roi_squares = data.create_dataset(path=data_path)	
	print 'Dataset successfully loaded.'
	return img_voxels,img_norm, mask_voxels,roi_squares

def load_empy_dataset(data_path):
	print 'loading data...'
	img_voxels,img_norm, mask_voxels,roi_squares = data.create_empty_dataset(path=data_path)
	print 'Dataset successfully loaded.'
	return img_voxels,img_norm, mask_voxels,roi_squares

model = load_model(filename = model_filename)

sys.stdout.flush()

img_voxels,img_norm, mask_voxels,roi_squares = load_empy_dataset(path_to_contours)

# img_voxel,_, _ ,_ = data.get_data('/home/ianben/Breast_MRI_cont_dcm/032/',data.get_contour_file('/home/ianben/Breast_MRI_cont_dcm/032/'), roi_name = 'GTV')
# img_voxel_arr = np.asarray(np.expand_dims(np.asarray(img_voxel), axis=3))

# img_norm,_, mask_voxels,roi_squares = load_data(data_path = path_to_contours)
sys.stdout.flush()

img_voxels = np.expand_dims(np.asarray(img_voxels), axis = 3)
# mask_voxels = np.asarray(mask_voxels)
# roi_squares = np.asarray(roi_squares)

print('~~~model.Predicting~~~')
sys.stdout.flush()

# img_norm_predicted = model.predict(img_norm,verbose=1)

img_voxels_predicted = model.predict(img_voxels,verbose=1)
print('~~~model.Predicted~~~\nSaving..')


img_voxels_predicted = np.squeeze(img_voxels_predicted, axis = 3)
img_voxels = np.squeeze(img_voxels,axis=3)


sys.stdout.flush()

# img_norm = np.squeeze(img_norm, axis = 3)
# img_norm_predicted = np.squeeze(img_norm_predicted, axis = 3)

# print ('img array shape:',img_norm.shape)
# print ('Roi arr shape: ',roi_squares.shape)
# print ('Predicted roi arr shape: ',img_norm_predicted.shape)

def create_images(img, roi, roi_pred, img_save_path, img_names):
	
	i=0
	for img,roi_good,roi_predicted in zip(img,roi,roi_pred):
		plt.subplot(1,3,1)
		plt.imshow(img,cmap = 'gray')
		plt.title('MRI')

		plt.subplot(1,3,2)
		plt.imshow(roi_good)
		plt.title('ROI good')

		plt.subplot(1,3,3)
		plt.imshow(roi_predicted)
		plt.title('ROI predicted')

		plt.savefig(img_save_path+img_names+str(i)+'.png')
		plt.close('all')
		i+=1
	return None
print('Creating Images..')
sys.stdout.flush()

create_images(img_voxels,roi_squares,img_voxels_predicted,img_save_path,img_names)

print ('done')
