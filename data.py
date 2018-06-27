#import dicom_contour.contour as dcm
#import pathlib as Path
import numpy as np
import matplotlib.pyplot as plt

import pydicom as dicom
import logging,warnings
import glob,os
import operator
import cv2

def listdir_nohidden(path):
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


logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='errors.log',level=logging.DEBUG)



def create_dataset(image_shape=512, n_set='train', path='./Breast_MRI_cont/', roi_name = 'GTV'):

    images, images_norm, contour_mask, roi_squares= [], [], [], []

    if path[-1] != '/': path += '/'

    #loop over all the files
    for filename in listdir_nohidden(path):
        if filename[-1] != '/': filename += '/'
        #get the contour file name
        contour_filename = get_contour_file(filename)

        img_voxel,img_norm_voxel, mask_voxel,roi_square_ = get_data(filename, contour_filename, roi_name = roi_name)

        images += img_voxel
        images_norm += img_norm_voxel
        contour_mask += mask_voxel
        roi_squares += roi_square_

    return images, images_norm, contour_mask,roi_squares


# In[5]:


#image_path = './Breast_MRI_cont/Belec/'
#contour_filename = 'RS.1.2.246.352.71.4.139189879485.219269.20180611142708.dcm'


#rt_sequence = dicom.read_file(image_path+contour_filename)
#print rt_sequence


# In[6]:


#dcm.get_roi_names(rt_sequence)
#type(dcm.get_roi_names(rt_sequence))

#y = rt_sequence.ROIContourSequence[1]
#contours = [contour for contour in y.ContourSequence]


# In[7]:

def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def get_roi_contour_ds(rt_sequence, index=-1, name = ''):
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

        # index 0 means that we are getting RTV information
        ROI = rt_sequence.ROIContourSequence[index]
        # get contour datasets in a list
        contours = [contour for contour in ROI.ContourSequence]
        return contours

    except (IndexError,ValueError), e:
        logging.debug('There was an error in <<get_roi_contour_ds>>.')
        #return an empty list
        return []


# In[8]:


#contour_datasets = get_roi_contour_ds(rt_sequence=rt_sequence, name = 'Orgn')
#print contour_datasets


def contour2poly(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
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

    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask



def get_mask_dict(contour_datasets, path):
    """
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
    #        print coords
            mask = poly_to_mask(coords, *shape)
    #        print('mask\n\n')
            img_contours_dict[img_id] += mask
        return img_contours_dict
    except TypeError, e:
        logging.debug('TypeError in <<get_mask_dict>>.')



def parse_dicom_file(filename):

    """
    Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
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
        return None


#the dictionary containing the slice -- mask information

#mask_dict = get_mask_dict(1, image_path)

#get the orders the slices come in
#slice_orders = dcm.slice_order(image_path)


#mask_dict.items()


def contour2poly1(contour_dataset, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
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

    pixel_coords = [( np.abs(np.ceil((x - origin_x) / x_spacing)), np.ceil((origin_y + y) / y_spacing))  for x, y, _ in coord]

    return pixel_coords, img_ID, img_shape


def get_img_mask_voxel(slice_orders, mask_dict, image_path):
    """
    Construct image and mask voxels

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
    return img_voxel, mask_voxel

#img_data, mask_data = get_img_mask_voxel(slice_orders, mask_dict, image_path)

def show_img_msk_fromarray(img_arr, msk_arr, square_arr = [], alpha=0.35, sz=7, cmap='inferno', save_path=None):

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

    msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)

    square = get_roi_square(img_arr, msk_arr)

#   box = create_roi_square(bbox2(msk_arr))

    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 3, 1)

    plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap=cmap, alpha=alpha)

    plt.subplot(1, 3, 2)
    plt.imshow(img_arr, cmap='gray')

    plt.subplot(1,3,3)
    plt.imshow(square_arr)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

def get_roi_square(image, contour, shape_out = 64):
    """
    Create a binary mask with ROI from contour.
    Extract the maximum square around the contour.
    :param image: input image (needed for shape only)
    :param contour: numpy array contour (d, 2)
    :return: numpy array mask ROI (shape_out, shape_out)
    """

    contour = np.asarray(contour)
    X_min, Y_min = contour[:,0].min(), contour[:,1].min()
    X_max, Y_max = contour[:,0].max(), contour[:,1].max()
    w = X_max - X_min
    h = Y_max - Y_min
    mask_roi = np.zeros(np.asarray(image).shape)
    if w > h :
        mask_roi[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
    else :
        mask_roi[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
    return cv2.resize(mask_roi, (shape_out, shape_out), interpolation = cv2.INTER_NEAREST)

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


def get_data(image_path, contour_filename, roi_name):
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
    try:

        # read dataset for contour
        rt_sequence = dicom.read_file(image_path+contour_filename)

        # get contour datasets with index idx
        contour_datasets = get_roi_contour_ds(rt_sequence, name = roi_name)
    #    print contour_datasets

        # construct mask dictionary
        mask_dict = get_mask_dict(contour_datasets, image_path)


        # get slice orders
        slice_orders = slice_order(image_path)


        # get image and mask data for patient
        img_voxel, mask_voxel  = get_img_mask_voxel(slice_orders, mask_dict, image_path)

#        masks = [mask for mask in mask_voxel]
        square_voxel = []
        for mask in mask_voxel:
            square_voxel.append(np.asarray(create_roi_square(bbox2(mask))))

        img_norm_voxel = normalize(img_voxel)

        """
        mean = np.mean(np.asarray(img_voxels))
        min_ = np.amin(np.asarray(img_voxels))
        max_ = np.amax(np.asarray(img_voxels))
        """

        return img_voxel,img_norm_voxel.tolist(), mask_voxel, square_voxel
    except OSError:
        logging.debug('OSErrir in <<get_data>>. Path: {}{}'.format(image_path,contour_filename))

def normalize(image):
    MIN_BOUND,MAX_BOUND = np.amin(np.asarray(image)),np.amax(np.asarray(image))

    mean = np.mean(np.asarray(image))
    std = np.std(np.asarray(image))

    img = image - mean
    img = img/std
    return img

def bbox2(img):
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
def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def create_roi_square(bbox = (0,0,0,0), orig_img_size = (512,512), output_size = (32,32)):
    """
    Create a grid of size output_size as
    Inputs:
        bbox:coordinates of a bounding box, ordered (row1,row2,col1,col2)
    """
    arr = np.zeros(orig_img_size)

    if bbox is (0,0,0,0):
        return np.meshgrid(arr)
    else:
        Y_min = bbox[0]
        Y_max = bbox[1]
        X_min = bbox[2]
        X_max = bbox[3]

        w = X_max - X_min
        h = Y_max - Y_min

        if w > h :
            arr[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
        else :
            arr[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
        return cv2.resize(arr, output_size, interpolation = cv2.INTER_NEAREST)
