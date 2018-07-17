import numpy as np
import matplotlib.pyplot as plt
import data
import cv2
import scipy.misc as msc


dict_cont,dict_img,dict_roi,imgs = data.load_png_from_file('./Breast_MRI_cont_png_1/001')


img_voxel,img_norm_voxel, mask_voxel,roi_square  = data.get_data('./Breast_MRI_test/001/',contour_filename = data.get_contour_file('./Breast_MRI_test/001'),roi_name='GTV')

# key_c, value_c = [(key,value) for key,value in sorted(dict_cont.iteritems(), key=lambda (k,v): (v,k))]
# key_i , value_i = [(key,value) for key,value in sorted(dict_img.iteritems(), key=lambda (k,v): (v,k))]
# key_r, value_r = [(key,value) for key,value in sorted(dict_roi.iteritems(), key=lambda (k,v): (v,k))]

assert(len(dict_cont)==len(dict_img)==len(dict_roi)), 'Not same length for some reason...'


# print '~~~~~~~~~CONTOURS~~~~~~~~~~~'
# for key, value in sorted(dict_cont.iteritems(), key=lambda (k,v): (v,k)):
# 	png_contour = plt.imread(key)
	
# 	msk_img = mask_voxel[value[0]]

# 	#print 'Shape orig: ', msk_img.shape, ' dict_shape: ',png_contour.shape
# 	if not np.allclose(png_contour[:,:,0],msk_img):
# 		print key


print '~~~~~~~~IMAGES~~~~~~~~~'

for key, value in sorted(dict_img.iteritems(), key=lambda (k,v): (v,k)):
	# png_img = plt.imread(key)

	#png_img = cv2.imread(key,0)	
	png_img = np.int16(scipy.misc(key,'I'))


	if not (np.allclose(png_img,img_voxel[value[0]]):
		print '\n',value
	
	plt.subplot(1,2,1)
	plt.imshow(img_)
	
	plt.subplot(1,2,2)
	plt.imshow(png_img)

	plt.savefig('/Users/ianbenlolo/Documents/Hospital/images/'+str(value[0])+'img.png')
	plt.close('all')
	
"""
print '~~~~~~~~ROIS~~~~~~~~~'

for key, value in sorted(dict_roi.iteritems(), key=lambda (k,v): (v,k)):
	png_roi = plt.imread(key)

	roi_ = roi_square[value[0]]


	if not np.allclose(png_roi[:,:,0],roi_):
		print key
"""
# 	plt.subplot(1,2,1)
# 	plt.imshow(mask_voxel[value[0]], cmap  = 'inferno')
	
# 	plt.subplot(1,2,2)
# 	plt.imshow(png_roi,cmap = 'inferno')

# 	plt.savefig('/Users/ianbenlolo/Documents/Hospital/images/'+str(value[0])+'img.png')




