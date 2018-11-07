'''
Created by Ian Benlolo. Mcgill Medical Physics dept. 
Some of these functions were adapted by Kerem Turgutlu's dicom-contour package. https://github.com/KeremTurgutlu/dicom-contour

'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydicom as dicom
import warnings
import glob,os
import operator
import sys,re
import cv2
from keras.preprocessing.image import ImageDataGenerator

def listdir_nohidden(path):
    '''
    Returns list of non-hidden files within directory
    '''
    return glob.glob(os.path.join(path, '*'))

def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    try:
        if path[-1] != '/': path += '/'

        # get .dcm contour file
        fpaths = [f for f in listdir_nohidden(path) if '.dcm' in f]

        n = 0
        contour_file = ''
        for fpath in fpaths:
            f = dicom.read_file(fpath)
            if 'ROIContourSequence' in dir(f):
                contour_file = fpath.split('/')[-1]
                n += 1
        if n > 1: warnings.warn("There are multiple files, returning the last one!")
        return contour_file
    except dicom.errors.InvalidDicomError:
        warnings.warn('File is missing DICOM File Meta Information header. Path:')
        print path
        return None


def create_dataset(path, roi_name = 'GTV', non_empty = False, mask_size = (32,32)):
    """
    This function extracts and returns all dicom files and contours within the given path directory.
    Inputs:
        path (str):  path of the the directory that has patient directories containg DICOM RT structs in it.
        roi_name: 'GTV' or 'Organ'
    """
    images, images_norm, contour_mask, roi_squares= [], [], [], []
    
    if path[-1] != '/': path += '/'

    #loop over all the files
    for filename in listdir_nohidden(path):
        if filename[-1] != '/': filename += '/'
        
        #get contour file name
        contour_filename = get_contour_file(filename)

        #get data
        img_voxel,img_norm_voxel, mask_voxel,roi_square = get_data(filename, contour_filename, roi_name = roi_name,mask_size=mask_size)
        if non_empty:
            for i,n,m,r in zip(img_voxel,img_norm_voxel, mask_voxel,roi_square):
                # if roi_square is not equal to all 0's we append all the data 
                if (not np.allclose(np.zeros(r[:,:,0].shape, dtype = r.dtype),r[:,:,0])):
                    if (i is None) or (r is None):
                        warnings.warn("None found!")
                    images.append(i)
                    images_norm.append(n)
                    contour_mask.append(m)
                    roi_squares.append(r)
        else: 
            images += img_voxel
            contour_mask += mask_voxel
            #commented since I wasnt using it anymore and it would increase memory by a lot
            images_norm += img_norm_voxel
            roi_squares += roi_square
        # print 'Loaded.'
    return images, images_norm, contour_mask,roi_squares

#y = rt_sequence.ROIContourSequence[1]
#contours = [contour for contour in y.ContourSequence]

def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the ROI's in dcm file
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def get_roi_contour_ds(rt_sequence, name = '',index= -1):
    """
    Extract desired ROI contour datasets
    from RT Sequence.

    E.g. rt_sequence can have contours for different parts of the brain
    such as ventricles, tumor, etc...

    You can use get_roi_names to find which index to use

    Inputs:
        rt_sequence (dicom.dataset.FileDataset): Contour file dataset, what you get
                                                 after reading contour DICOM file
        index (int): Index for ROI Sequence
    Return:
        contours (list): list of ROI contour dicom.dataset.Dataset s
    """

    try:
        #get the actual index it is at
        if (index == -1):
            roi_indeces = get_roi_names(rt_sequence)
            index = roi_indeces.index(name)

        ROI = rt_sequence.ROIContourSequence[index]
        # get contour datasets in a list
        contours = [contour for contour in ROI.ContourSequence]
        return contours

    except (IndexError,ValueError), e:
        warnings.warn('There was an error in <<get_roi_contour_ds>>.Returning empty array')
        return []
    except (AttributeError), e:
        return []


def contour2poly(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs:
        contour_dataset (dicom.dataset.Dataset) : DICOM dataset class that is identified as
                         (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(path +'MR.'+ img_ID + '.dcm')
    img_arr = img.pixel_array
    img_shape = img_arr.shape

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [( np.ceil(np.abs((x - origin_x) / x_spacing)), np.abs(np.ceil((origin_y - y) / y_spacing))  ) for x, y, _ in coord]

    return pixel_coords, img_ID, img_shape



def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw
    """
    Convert polygon to mask
    Inputs:
        polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
            in units of pixels
        width: scalar image width
        param height: scalar image height
    Return: 
        Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.int8(np.array(img).astype(bool))
    return mask



def get_mask_dict(contour_datasets, path):
    """
    Get a dictionary of (Image ID: Contour array).
    Inputs:
        contour_datasets (list): list of dicom.dataset.Dataset for contours
        path (str): path of directory with images
    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """
    try:
        from collections import defaultdict

        # create empty dict for
        img_contours_dict = defaultdict(int)

        for cdataset in contour_datasets:
            coords, img_id, shape = contour2poly(cdataset, path)
            
            mask = poly_to_mask(coords, *shape)

            img_contours_dict[img_id] += mask
        return img_contours_dict
    except TypeError, e:
        warnings.warn('TypeError in <<get_mask_dict>>.')



def parse_dicom_file(filename):

    """
    Parse the given DICOM filename.
    Input:
        filename: filepath to the DICOM file to parse
    Return: 
        Dictionary with DICOM image data
    """
    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        return dcm_image
    except dicom.errors.InvalidDicomError:
        warnings.warn('InvalidDicomError in <<parse_dicom_file>>')
        return None




def contour2poly1(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Input:
        contour_dataset (dicom.dataset.Dataset) : DICOM dataset class that is identified as
                         (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(path +'MR.'+ img_ID + '.dcm')
    img_arr = img.pixel_array
    img_shape = img_arr.shape

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient
    # y, x is how it's mapped

    pixel_coords = [(np.abs(np.ceil((x - origin_x) / x_spacing)), np.ceil((origin_y + y) / y_spacing)) for x, y, _ in coord]

    return pixel_coords, img_ID, img_shape


def get_img_mask_voxel(slice_orders, mask_dict, image_path):
    """
    Construct image and mask voxels.

    Inputs:
        slice_orders (list): list of tuples of ordered img_id and z-coordinate position
        mask_dict (dict): dictionary having img_id : contour array pairs
        image_path (str): directory path containing DICOM image files
    Return:
        img_voxel: ordered image voxel for CT/MR
        mask_voxel: ordered mask voxel for CT/MR
    """

    img_voxel = []
    mask_voxel = []

    for img_id, _ in slice_orders:
        img_array = parse_dicom_file(image_path + 'MR.'+img_id + '.dcm')
        if mask_dict is not None:
            if img_id in mask_dict:
                mask_array = mask_dict[img_id]
            else:
                mask_array = np.zeros_like(img_array)
            img_voxel.append(img_array)
            mask_voxel.append(mask_array)
        else:
            img_voxel.append(img_array)
            mask_voxel.append(np.zeros_like(img_array))
    return img_voxel, mask_voxel


def show_img_msk_fromarray(img_arr, msk_arr, square_arr = None, alpha=0.35, sz=7, cmap='inferno', save_path=None):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """

    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path+'_img.png',bbox_inches='tight')
    plt.close('all')

    plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.savefig(save_path+'_img_mask.png',bbox_inches='tight')
    plt.close('all')

    plt.imshow(msk_arr, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path+'_msk.png',bbox_inches='tight')
    plt.close('all')

    # plt.figure(figsize=(sz, sz))
    # if square_arr is not None:
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(img_arr, cmap='gray')
    #     plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    #     plt.title("Img with Mask")
    #     plt.axis('off')

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(img_arr, cmap='gray')
    #     plt.title("Image")
    #     plt.axis('off')

    #     plt.subplot(1,3,3)
    #     plt.imshow(square_arr)
    # else:
    #     plt.subplot(1,2,1)
    #     plt.imshow(img_arr, cmap='gray')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     # plt.imshow(img_arr, cmap='gray')
    #     plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    #     plt.axis('off')

    # if save_path is None:
    #     plt.show()
    # else:
    #     plt.savefig(save_path)
    #     plt.close('all')
        

def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue
    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_data(image_path, contour_filename, roi_name='GTV',mask_size = (32,32)):
    """
    Given image_path, contour_filename and roi_index return
    image and mask voxel array

    Inputs:
        image_path (str): directory path containing DICOM image files
        contour_filename (str): absolute filename for DICOM contour file
        roi_index (int): index for desired ROI from RT Struct
    Return:
        img_voxel (np.array): 3 dimensional numpy array of ordered images
        mask_voxel (np.array): 3 dimensional numpy array of ordered masks
    """
    if image_path[-1] != '/': image_path += '/'
    try:
        # read dataset for contour
        try:
            if contour_filename is not None:
                rt_sequence = dicom.read_file(image_path+contour_filename)
            else:
                raise dicom.errors.InvalidDicomError()
            # get contour datasets name roi_name
            contour_datasets = get_roi_contour_ds(rt_sequence, name = roi_name)

            if not contour_datasets:
                print 'No contour.  Image path: ',image_path
            # construct mask dictionary
            mask_dict = get_mask_dict(contour_datasets, image_path)

        except dicom.errors.InvalidDicomError:
            warnings.warn('There was an error opening the dicom file. path:')
            print image_path

            rt_sequence = None
            contour_datasets = None
            mask_dict = None

        # get slice orders
        slice_orders = slice_order(image_path)

        # get image and mask data for patient
        img_voxel, mask_voxel  = get_img_mask_voxel(slice_orders, mask_dict, image_path)

        square_voxel = []
        for mask in mask_voxel:
            #create roi square mask
            roi = create_roi_square(bbox=bbox(mask),output_size = mask_size)

            #to add that extra dimension since keras requires one hot encoding.
            roi = create_3rd_dim(roi)

            #convert to int to make easier on memory
            square_voxel.append(np.int8(np.asarray(roi)))

        img_norm_voxel = normalize(img_voxel)

        return img_voxel,img_norm_voxel.tolist(), mask_voxel, square_voxel

    except OSError:
        path = image_path+contour_filename
        warnings.warn('OSError in <<get_data>>. Path:')
        print path

def create_3rd_dim(roi):
    """
    Creates a second layer to the masks since keras requires one-hot encoded. Trivial but necessary.
    Inputs:
        roi (np array): a 32x32 mask 
    Return:
        ret_roi : a 32x32x2 mask where the second layer is simply the binary opposite of the input one (and the first is the same)
    """
    ret_roi = np.empty((2 ,roi.shape[0], roi.shape[1]))

    ret_roi[0] = roi
    ret_roi[1] = np.absolute(np.ones(roi.shape)-roi)
    return np.moveaxis(ret_roi,0,-1)

def normalize(image):
    """
    Center and normalize image
    """
    if image is None:
        return none

    MIN_BOUND,MAX_BOUND = np.amin(np.asarray(image)),np.amax(np.asarray(image))

    mean = np.mean(np.asarray(image))
    std = np.std(np.asarray(image))

    img = np.asarray(image) - mean
    img = np.asarray(img)/std
    return img
def bbox(img):
    """
    Get a bounding box around an image
    Input:
        img: the image (the binary mask)
    Output:
        the coordinates of the bounding box
    """
    try:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax
    except:
        return (0,0,0,0)

def create_roi_square(bbox = (0,0,0,0), orig_img_size = (512,512), output_size = (32,32)):
    """
    Create a grid of size output_size as
    Inputs:
        :bbox:coordinates of a bounding box, ordered (row1,row2,col1,col2)
    Outputs:
        A binary (output_size) mask with a square region covering the roi
    """
    
    if (bbox==(0,0,0,0)):#(bbox[0]==0&bbox[1]==0&bbox[2]==0&bbox[3]==0):
        return np.zeros(output_size)
    else:
        arr = np.zeros(orig_img_size)
        Y_min = bbox[0]
        Y_max = bbox[1]
        X_min = bbox[2]
        X_max = bbox[3]

        w = (X_max - X_min)
        h = (Y_max - Y_min)
        
        # X_min -= w/2
        # X_max += w/2
        # Y_min -= h/2
        # Y_max += h/2
        # w = (X_max - X_min)
        # h = (Y_max - Y_min)

        if w > h :
            arr[int(Y_min - (w - h)/2):int(Y_max + (w - h)/2), int(X_min):int(X_max)] = 1.0
        else :
            arr[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
        return cv2.resize(arr, output_size, interpolation = cv2.INTER_NEAREST)

# def get_roi_square(image, contour, shape_out = 64):
#     """
#     Create a binary mask with ROI from contour.
#     Extract the maximum square around the contour.
#     Input:
#         image: input image (needed for shape only)
#         contour: numpy array contour (d, 2)
#     return: numpy array mask ROI (shape_out, shape_out)
#     """

#     contour = np.asarray(contour)
#     X_min, Y_min = contour[:,0].min(), contour[:,1].min()
#     X_max, Y_max = contour[:,0].max(), contour[:,1].max()
#     w = 2.*(X_max - X_min)
#     h = 2.*(Y_max - Y_min)
#     mask_roi = np.zeros(np.asarray(image).shape)
#     if w > h :
#         mask_roi[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
#     else :
#         mask_roi[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
#     return cv2.resize(mask_roi, (shape_out, shape_out), interpolation = cv2.INTER_NEAREST)

def load_png_from_file(directory):
    if directory[-1] != '/': directory += '/'

    #open the directory
    imgs = listdir_nohidden(directory)
    imgs.sort()

    #find matches to */XXX/YYY_contour.png or _img.png or _roi.png 
    cont_ = [re.match(r'\./{0,1}.*/?[0-9]{3}/(?P<number>.*)_contour.png',img)  for img in imgs]
    img_ = [re.match(r'\./{0,1}.*/?[0-9]{3}/(?P<number>.*)_img.png',img) for img in imgs]
    roi_ = [re.match(r'\./{0,1}.*/?[0-9]{3}/(?P<number>.*)_roi.png',img) for img in imgs]

    #make a dictionary for each and return it 
    dict_cont= dict()
    dict_img = dict()
    dict_roi = dict()
    for cont_match,img_match,roi_match,img in zip(cont_,img_,roi_,imgs):
        if cont_match is not None:
            dict_cont.setdefault(img, []).append(int(cont_match.group('number')))
        elif img_match is not None:
            dict_img.setdefault(img, []).append(int(img_match.group('number')))
        elif roi_match is not None:
            dict_roi.setdefault(img, []).append(int(roi_match.group('number')))
        else:
            print 'Could not find match for ', img
    return dict_cont,dict_img,dict_roi,imgs


def discretize(mask, cutoff=0.9):
    mask[np.where(mask > cutoff)] = 1
    mask[np.where(mask < cutoff)] = 0
    return mask

def dataset(save_path = '/home/ianben/', path_to_contours = '/home/ianben/Breast_MRI_dcm/',non_empty = False):
    print 'train: '
    train = create_dataset(path = path_to_contours+'train/', non_empty = non_empty)
    print np.asarray(train).shape, type(train)
    np.save(save_path + 'train.npy',train)
    train = []

    print 'test: '
    test = create_dataset(path = path_to_contours+'test/', non_empty = non_empty)
    np.save(save_path + 'test.npy',test)
    
    
def trainGenerator(batch_size, train_path,aug_dict, image_folder = 'images', mask_folder='contours', image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    flag_multi_class = False, num_class = 2, save_to_dir = None, target_size = (512,512), seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


if __name__=='__main__':
    path_to_contours = '/Users/ianbenlolo/Documents/Hospital/test/'

    # data= create_dataset(path = path_to_contours, non_empty = True) 
    # np.save('./data.npy',np.asarray(data))  
    # dataset(non_empty = True)
    img_voxel,norm, mask_voxel,square = np.load('./train.npy')

    print img_voxel.shape,norm.shape,np.asarray(mask_voxel).shape,np.asarray(square).shape
    # i=0

    # print 'Creating images'
    # for img, mask,sq in zip(img_voxel, mask_voxel,square):
    #     show_img_msk_fromarray(img_arr = img, msk_arr = mask[:,:],square_arr = sq[:,:,0], sz=10, cmap='inferno', alpha=0.65,save_path = '/Users/ianbenlolo/Documents/Hospital/images/'+str(i))
    #     i+=1
