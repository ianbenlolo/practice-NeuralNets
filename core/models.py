import keras
from keras import optimizers,regularizers
from keras import backend as K
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.utils import plot_model

def create_model(activ='relu', input_shape = (512,512)):
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(input_shape[0], input_shape[1],1)))
	model.add(Conv2D(filters = 32,
		kernel_size = (12,12),
		activation = activ,
		strides=(1, 1),
		padding='valid'
		#input_shape=(input_shape[0], input_shape[1],1)
		))
	model.add(MaxPooling2D(pool_size=(12,12)))
	model.add(Conv2D(filters = 64,
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
	model.add(Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
	model.add(Reshape([32,32,2]))
	return model

def create_deep_model(activ='relu', input_shape = (512,512)):
	"""
	Create deep model. 
	Input:
		Activ: activation function for EVERY layer.
	Return:
		The model
	"""
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid', input_shape=(input_shape[0], input_shape[1],1)))
	model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(1,1)))

	model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))	
	model.add(MaxPooling2D(pool_size=(4,4)))

	model.add(Flatten())
	model.add(Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
	model.add(Reshape([32,32,2]))
	return model

def create_deep_model_with_normalization(activ='relu', input_shape = (512,512),dropout=0):
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid', input_shape=(input_shape[0], input_shape[1],1)))
	model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(BatchNormalization())
	# model.add(Dropout(dropout))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(BatchNormalization())
	# model.add(Dropout(dropout))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters = 512, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(Conv2D(filters = 512, kernel_size = (3,3), activation = activ, strides=(1, 1), padding='valid'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Flatten())
	model.add(Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
	model.add(Reshape([32,32,2]))
	return model
	
# def unet(pretrained_weights = None,input_size = (512,512,1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     # drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#     # drop5 = Dropout(0.5)(conv5)

#     up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
#     merge6 = concatenate([conv4,up6],axis=3) #was drop4 instead of conv4
#     conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

#     up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     merge7 = concatenate([conv3,up7],axis=3)
#     conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

#     up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     merge8 = concatenate([conv2,up8],axis=3)
#     conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

#     up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     merge9 = concatenate([conv1,up9],axis=3)
#     conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

#     model = Model(input = inputs, output = conv10)
#     return model

def unet(pretrained_weights = None, input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    #model.summary()
    return model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

if __name__=='__main__':
	model = create_deep_model_with_normalization()
	# for p in m.layers:
	# 	print p.name.title(), p.input_shape, p.output_shape
	# print 
	# m.summary()
	
	plot_model(model, to_file = 'model.png',show_shapes=True)
