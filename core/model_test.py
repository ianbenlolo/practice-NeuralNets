'''
Created by Ian Benlolo. Mcgill Medical Physics dept. 
'''
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Model
import data,sys
import os

model_name = '37_2'
path_to_contours = '/home/ianben/test.npy'
model_data_name = 'u'

RUN_ONE_IMAGE = False
SLICE = 31
patient_id = '151'


# def get_data(path,non_empty = False):
# 	if non_empty:
# 		return data.create_empty_dataset(path=data_path)
# 	else:
# 		return np.load('/home/ianben/full_dataset.npy')

def create_images_mask(img, roi, roi_pred,img_save_path):
	i=0
	for img,roi_good,roi_predicted in zip(img,roi,roi_pred):
		plt.subplot(1,3,1)
		plt.imshow(img,cmap = 'gray')
		plt.title('MRI')
		plt.axis('off')

		plt.subplot(1,3,2)
		plt.imshow(roi_good, cmap = 'gray')
		plt.title('ROI good')
		plt.axis('off')

		# plt.subplot(1,5,3)
		# plt.imshow(roi_predicted)
		# plt.title('ROI predicted')
		# plt.axis('off')
		
		# plt.subplot(1,4,3)
		# plt.imshow(roi_predicted,cmap = 'gray')
		# plt.title('ROI gray')
		# plt.axis('off')

		plt.subplot(1,3,3)
		plt.imshow(data.discretize(roi_predicted))
		plt.title('ROI discret.')
		plt.axis('off')

		save = img_save_path+str(i)+'.png'
		plt.savefig(save)
		plt.close('all')
		i+=1
	return None
def create_images_contour(img,cont,cont_pred,img_save_path):
	i=0
	for img,cont_good,cont_predicted in zip(img,cont,cont_pred):
		plt.subplot(1,4,1)
		plt.imshow(img,cmap = 'gray')
		plt.title('MRI')
		plt.axis('off')

		plt.subplot(1,4,2)
		plt.imshow(cont_good, cmap = 'gray')
		plt.title('Cont good')
		plt.axis('off')

		plt.subplot(1,4,3)
		plt.imshow(cont_predicted,cmap = 'gray')
		plt.title('Cont predicted')
		plt.axis('off')
		
		plt.subplot(1,4,4)
		plt.imshow(data.discretize(cont_predicted), cmap = 'gray')
		plt.title('ROI discret.')
		plt.axis('off')

		save = img_save_path+str(i)+'.png'
		plt.savefig(save)
		plt.close('all')
		i+=1
	return None

def run():
	img_save_path = '/home/ianben/images/test_'+model_name+'/'

	print 'Loading model: ',model_name

	if RUN_ONE_IMAGE:
		print 'patient: ',patient_id,' Slice: ',SLICE

	#make sure save path exists
	if not os.path.exists(img_save_path): os.mkdir(img_save_path)

	print('-'*30)
	#load model which is a .hd5 file
	model = keras.models.load_model('/home/ianben/models/'+ model_name+ '.h5')
	model.load_weights('/home/ianben/checkpoints/model_weights_'+model_name+'.hdf5')

	print 'Creating dataset..'
	if RUN_ONE_IMAGE:
		path='/home/ianben/Breast_MRI_dcm/train/' + patient_id + '/'

		_,img_voxels, mask_voxel ,_ = data.get_data(path,data.get_contour_file(path))

		#get required slice
		img_voxels = img_voxels[SLICE]
		mask_voxel = mask_voxel[SLICE]
		# roi_squares = roi_squares[SLICE]

		#copy it 10 times for testing purposes
		mask_voxel = np.tile(np.asarray(mask_voxel),(10,1,1))
		img_voxels = np.tile(np.asarray(img_voxels),(10,1,1))

		# roi_squares = np.asarray(np.tile(roi_squares,(10,1,1,1)))
		roi_squares=np.zeros((1,1))

		plt.subplot(1,2,1)
		plt.imshow(img_voxels[0], cmap = 'gray')
		plt.subplot(1,2,2)
		plt.imshow(mask_voxel[0],cmap = 'gray')

		plt.savefig('./img_test.png')
		plt.close('all')
	else:
		# img_voxels,_, mask_voxel ,roi_squares = data.create_dataset(path = path_to_contours)
		_,img_voxels, mask_voxel,roi_squares = np.load(path_to_contours)
		img_voxels=np.asarray(img_voxels)
		mask_voxel = np.asarray(mask_voxel)
		roi_squares = np.asarray(roi_squares)
	

	img_voxels = np.expand_dims(img_voxels, axis = 3)

	print 'Done.'
	print 'img_voxels shape: ', img_voxels.shape, ' roi_squares shape: ', roi_squares.shape, ' mask_voxel shape ', mask_voxel.shape
	print '~~~~model.Predicting~~~~'
	
	def chunks(l, n):
		for i in range(0, len(l), n):
			yield l[i:i + n]

	if model_data_name == 'u':
		# create_images_contour(img_voxels,mask_voxel,img_mask_pred,img_save_path)
		# mask_voxel_pred = model.predict(np.asarray(img_voxels),verbose=1)
		mask_voxel_pred =  np.squeeze(np.vstack([model.predict(np.asarray(img),verbose=1) for img in chunks(img_voxels,7)]),axis=3)
		img_voxels = np.squeeze(img_voxels,axis=3)

		print 'Creating Images. Save path: ',img_save_path
		print img_voxels.shape,mask_voxel.shape, mask_voxel_pred.shape
		create_images_contour(img_voxels,mask_voxel,mask_voxel_pred,img_save_path)
	else:
		img_mask_pred = np.vstack([model.predict(np.asarray(img),verbose=1) for img in chunks(img_voxels,7)])

		img_voxels = np.squeeze(img_voxels,axis=3)

		print 'Creating Images.. Save path: ',img_save_path
		print np.asarray(img_voxels).shape, np.asarray(roi_squares).shape ,np.asarray(img_mask_pred).shape 

		create_images_mask(img_voxels,roi_squares[:,:,:,0],img_mask_pred[:,:,:,0],img_save_path)
if __name__ == '__main__':
	run()
	print 'Done.\n\n'


