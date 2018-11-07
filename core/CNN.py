'''
Created by Ian Benlolo. Mcgill Medical Physics dept. Summer 2018.
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras import Model
from keras import optimizers,regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
import sys,pickle
import models
import data

runName = '32'
batch_size = 8
epochs = 20
MODEL_ = 'u'
data_augm = False
validation_split = 0.2
dropout_rate = 0.1
on_MPU = True
loss = models.dice_coef_loss #'mean_squared_error'


#if running only with one image set to true
RUN_ONE_IMAGE = False
patient = '151'
SLICE = 31
COPIES = 1000

def dataset(non_empty = False):
	data.dataset(non_empty = non_empty)

def create_model(model):
	'''
	Input: model "key" 
	'r' for regular model
	'd' for deep model
	'dn' for deep model with batch norm
	'u' for U-net 
	See models.py for full models
	'''
	m=1
	if model == 'r':
		m = models.create_model()
	elif model == 'd':
		m = models.create_deep_model()
	elif model =='dn': 
		m = models.create_deep_model_with_normalization(dropout = dropout_rate)
	elif model=='u':
		m = models.unet()

	print 'Size for each layer :\nLayer, Input Size, Output Size'
	for p in m.layers:
	   print p.name.title(), p.input_shape, p.output_shape
	print '\nModel summary:'
	m.summary()
	return m
# class MultiGPUCheckpointCallback(Callback):
# 	def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
# 				 save_best_only=False, save_weights_only=False,
# 				 mode='auto', period=1):
# 		super(MultiGPUCheckpointCallback, self).__init__()
# 		self.base_model = base_model
# 		self.monitor = monitor
# 		self.verbose = verbose
# 		self.filepath = filepath
# 		self.save_best_only = save_best_only
# 		self.save_weights_only = save_weights_only
# 		self.period = period
# 		self.epochs_since_last_save = 0

# 		if mode not in ['auto', 'min', 'max']:
# 			warnings.warn('ModelCheckpoint mode %s is unknown, '
# 						  'fallback to auto mode.' % (mode),
# 						  RuntimeWarning)
# 			mode = 'auto'

# 		if mode == 'min':
# 			self.monitor_op = np.less
# 			self.best = np.Inf
# 		elif mode == 'max':
# 			self.monitor_op = np.greater
# 			self.best = -np.Inf
# 		else:
# 			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
# 				self.monitor_op = np.greater
# 				self.best = -np.Inf
# 			else:
# 				self.monitor_op = np.less
# 				self.best = np.Inf

# 	def on_epoch_end(self, epoch, logs=None):
# 		logs = logs or {}
# 		self.epochs_since_last_save += 1
# 		if self.epochs_since_last_save >= self.period:
# 			self.epochs_since_last_save = 0
# 			filepath = self.filepath.format(epoch=epoch + 1, **logs)
# 			if self.save_best_only:
# 				current = logs.get(self.monitor)
# 				if current is None:
# 					warnings.warn('Can save best model only with %s available, '
# 								  'skipping.' % (self.monitor), RuntimeWarning)
# 				else:
# 					if self.monitor_op(current, self.best):
# 						if self.verbose > 0:
# 							print('Epoch %05d: %s improved from %0.5f to %0.5f,'
# 								  ' saving model to %s'
# 								  % (epoch + 1, self.monitor, self.best,
# 									 current, filepath))
# 						self.best = current
# 						if self.save_weights_only:
# 							self.base_model.save_weights(filepath, overwrite=True)
# 						else:
# 							self.base_model.save(filepath, overwrite=True)
# 					else:
# 						if self.verbose > 0:
# 							print('Epoch %05d: %s did not improve' %
# 								  (epoch + 1, self.monitor))
# 			else:
# 				if self.verbose > 0:
# 					print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
# 				if self.save_weights_only:
# 					self.base_model.save_weights(filepath, overwrite=True)
# 				else:
# 					self.base_model.save(filepath, overwrite=True)
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def training(m, X, Y, verbose=2, batch_size=8, epochs= 10, validation = 0.1,data_aug=False):
    """
    Training CNN with the possibility to use data augmentation
    :param m: Keras model
    :param X: training pictures
    :param Y: training binary ROI mask
    :return: history
    """
    checkpoints_fp = '/home/ianben/checkpoints/model_weights_'+runName+'.hdf5'
    model_check=ModelCheckpoint(filepath = checkpoints_fp, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir='/home/ianben/logs/{}'.format(time()))
    callbacks_list = [model_check,tensorboard]

    if data_augm:
    	datagen_dict = dict(featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

        image_datagen = ImageDataGenerator(**datagen_dict)
        mask_datagen = ImageDataGenerator(**datagen_dict)

        seed = 1
        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(Y, augment=True, seed=seed)

        image_generator = image_datagen.flow(X, batch_size=batch_size)
        mask_generator = mask_datagen.flow(Y, batch_size=batch_size)

        history = m.fit_generator(zip(image_generator, mask_generator),
                                    steps_per_epoch=X.shape[0] // batch_size,
                                    epochs=epochs)         
    else:
        history = m.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation, shuffle = True, callbacks=callbacks_list)
    return history

def run():
	print 'runname: ', runName, ' Batch size: ', batch_size, ' epochs: ', epochs, ' dropout_rate ',dropout_rate, 'loss ',loss
	if RUN_ONE_IMAGE:
		print 'patient: ', patient, 'slice: ', SLICE, ' copies: ', COPIES

	path_to_contours = '/home/ianben/train.npy'
		# path_to_contours = '/home/ianben/Breast_MRI_dcm/train/'

	#Configuration. Not sure why needed but it is for multi processing on gpu's
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	K.set_session(sess)

	if RUN_ONE_IMAGE:
		path='/home/ianben/Breast_MRI_dcm/train/'+ patient + '/'

		_,img_voxel, roi_squares,_ = data.get_data(path,data.get_contour_file(path))

		#take required slice
		img_voxel = img_voxel[SLICE]
		roi_squares = roi_squares[SLICE]

		plt.subplot(1,2,1)
		plt.imshow(img_voxel, cmap = 'gray')
		plt.subplot(1,2,2)
		plt.imshow(roi_squares,cmap = 'gray')

		plt.savefig('img.png')
		plt.close('all')

		#copy it COPIES times
		img_voxel_arr = np.tile(img_voxel,(COPIES,1,1))

		roi_squares_voxels_arr = np.tile(roi_squares,(COPIES,1,1))
		roi_squares_voxels_arr = np.expand_dims(roi_squares_voxels_arr,3)
		img_voxel_arr = np.expand_dims(np.asarray(img_voxel_arr),3)

	else: # if NON_EMPTY_DATA:
		# 	img_voxel,_, _ ,roi_squares = data.create_non_empty_dataset(path = path_to_contours)
		# else:
		# 	# img_voxel,_, _ ,roi_squares = data.create_dataset(path = path_to_contours)
		# 	img_voxel,_, _ ,roi_squares = np.load('/home/ianben/full_dataset.npy')
		
		#if u-net were gna load the contours and not the square roi's
		if MODEL_ =='u':
			_,img_voxel, roi_squares ,_ = np.load(path_to_contours)
			roi_squares_voxels_arr = np.expand_dims(np.asarray(roi_squares),axis=3)
		else:
			_,img_voxel, _, roi_squares = np.load(path_to_contours)
			roi_squares_voxels_arr = np.asarray(roi_squares)

		#training expects a (512,512,1) img and msk
		img_voxel_arr = np.expand_dims(np.asarray(img_voxel),axis=3)


	print 'img array shape: ',img_voxel_arr.shape ,', square mask array shape: ',roi_squares_voxels_arr.shape
	
	print 'Creating model'
	m = create_model(MODEL_)

	print '____FITTING____'
	if on_MPU:
		

		# checkpoints=MultiGPUCheckpointCallback(checkpoints_fp, m, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		gpu_model = ModelMGPU(m, 2)

		gpu_model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
		history = training(gpu_model,img_voxel_arr,roi_squares_voxels_arr,validation = validation_split, data_aug = data_augm, batch_size=batch_size, epochs= epochs)
	# else:
	#     m.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
	#     history = training(m, img_voxel_arr,roi_squares_voxels_arr,callbacks = callbacks_list, batch_size=batch_size, epochs= epochs)

	#     with open('~/Users/ianbenlolo/Documents/models/'+runname+'_hist', 'wb') as file_pi:
	# 	   pickle.dump(history.history, file_pi)

	#clear mem
	img_voxel_arr = []
	roi_squares_voxels_arr = []

	m.save('/home/ianben/models/'+runName+'.h5')
	print 'Done.\n\n\n'

	import model_test as tst
	tst.model_name = runName
	tst.model_data_name = MODEL_

	tst.RUN_ONE_IMAGE = RUN_ONE_IMAGE
	tst.SLICE = SLICE
	tst.patient_id = patient
	

	tst.run()
if __name__=='__main__':
	m = create_model('dn')
