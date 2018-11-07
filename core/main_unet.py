import data
import models
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras import backend as K

path_to_contours = '/home/ianben/Breast_MRI_save/'

runName = '32'
MPU = True

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_path = path_to_contours + 'train/'
myGene = data.trainGenerator(batch_size = 2,train_path = train_path ,aug_dict = data_gen_args, save_to_dir = None)

model = models.unet()
model.summary()

model_checkpoint = ModelCheckpoint('unet_checkpoints.hdf5', monitor='loss',verbose=1, save_best_only=True)

if MPU:
	#set up multi gpu processing
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	K.set_session(sess)
	gpu_model = keras.utils.multi_gpu_model(model,gpus = 2)
	gpu_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr = 1e-4), metrics=['accuracy'])
	
	gpu_model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
else:
	model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr = 1e-4), metrics=['accuracy'])
	model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])


testGene = testGenerator(path_to_contours+'test/images/')

m.save('/home/ianben/models/'+runName+'.h5')
"""
results = model.predict_generator(testGene,30,verbose=1)

def create_images(img, roi, roi_pred,img_save_path):
	i=0
	for img,roi_good,roi_predicted in zip(img,roi,roi_pred):
		plt.subplot(1,5,1)
		plt.imshow(img,cmap = 'gray')
		plt.title('MRI')
		plt.axis('off')

		plt.subplot(1,5,2)
		plt.imshow(roi_good[:,:,0], cmap = 'gray')
		plt.title('ROI good')
		plt.axis('off')

		plt.subplot(1,5,3)
		plt.imshow(roi_predicted[:,:,0])
		plt.title('ROI predicted')
		plt.axis('off')
		
		plt.subplot(1,5,4)
		plt.imshow(roi_predicted[:,:,0],cmap = 'gray')
		plt.title('ROI gray')
		plt.axis('off')

		plt.subplot(1,5,5)
		plt.imshow(data.discretize(roi_predicted[:,:,0]))
		plt.title('ROI discret.')
		
		save = img_save_path+str(i)+'.png'
		plt.savefig(save)
		plt.close('all')
		i+=1
"""

