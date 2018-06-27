import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np

import keras
from keras.models import Model

import data

img_save_path = '/home/ianben/images/'
#cmap='inferno'
filename = 'first.h5'

#    img_arr, msk_arr, square_arr = [], alpha=0.35, sz=7, 

print 'Loading model..'
model = keras.models.load_model('/home/ianben/models/'+ filename)
print 'Model successfully loaded.'

print 'loading dataset..'
img_voxels,img_norm, mask_voxels,roi_squares = data.create_dataset(path='./Breast_MRI_cont/')

print 'Dataset successfully loaded.'

img_voxels_arr = np.expand_dims(np.asarray(img_voxels), axis=0)
roi_squares_voxels_arr = np.expand_dims(roi_squares,axis = 0)

img_voxels_arr = np.moveaxis(img_voxels_arr,0,-1)
roi_squares_voxels_arr_GOOD = np.moveaxis(roi_squares_voxels_arr,0,-1)


print 'Predicting...'
roi_squares_voxels_arr_PREDICTED = model.predict(img_voxels_arr,verbose=1)
print '~~~~Predicted~~~~~'

img_voxels_arr = np.moveaxis(img_voxels_arr,-1,0)
roi_squares_voxels_arr_GOOD = np.moveaxis(roi_squares_voxels_arr_GOOD,-1,0)
roi_squares_voxels_arr_PREDICTED = np.moveaxis(roi_squares_voxels_arr_PREDICTED,-1,0)
print ('img array shape:',img_voxels_arr.shape)
print ('Square roi array GOOD shape:',roi_squares_voxels_arr_GOOD.shape)
print ('Predicted roi arr shape: ',roi_squares_voxels_arr_PREDICTED.shape)

print ('Creating images..')
i=0
for img,roi_good,roi_predicted in zip(np.squeeze(img_voxels_arr),np.squeeze(roi_squares_voxels_arr_GOOD),np.squeeze(roi_squares_voxels_arr_PREDICTED)):
	fig, (ax1,ax2,ax3)= plt.subplots(1,3,sharex=True)
	ax1.imshow(img,cmap='gray')

	ax2.imshow(roi_good,cmap = 'inferno')

	ax3.imshow(roi_predicted,cmap = 'inferno')

	fig.savefig(img_save_path+'test'+str(i)+'.png')
	print ('image',str(i),'saved')
	i+=1
