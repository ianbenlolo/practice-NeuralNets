import CNN

NON_EMPTY = True
run = '40'
epochs =15
model = 'u'

data_augm = False
one_image = False

print '-'*50
print 'run ',run, ', epochs: ',epochs, ', Non empty data: ', NON_EMPTY, ' model: ',model,'\n','-'*30
print '-'*50
#create the train and test data set and save as .npy file in home dir.

CNN.dataset(non_empty = NON_EMPTY)

for i in (2,4,8,16,32,64):
	CNN.runName = run + '_%d'%i
	CNN.batch_size = i
	CNN.epochs = epochs
	CNN.MODEL_ = model
	CNN.data_augm = data_augm
	CNN.RUN_ONE_IMAGE = one_image

	CNN.run()