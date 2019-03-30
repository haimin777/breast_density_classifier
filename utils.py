import numpy as np
from scipy import misc
import pydicom as pyd
import cv2


def histogram_features_generator(image_batch, parameters):
    """
    Generates features for histogram-based model
    :param image_batch: list of 4 views
    :param parameters: parameter dictionary
    :return: array of histogram features
    """

    histogram_features = []
    x = [image_batch[0], image_batch[1], image_batch[2], image_batch[3]]

    for view in x:
        hist_img = []

        for i in range(view.shape[0]):
            hist_img.append(histogram_generator(view[i], parameters['bins_histogram']))

        histogram_features.append(np.array(hist_img))

    histogram_features = np.concatenate(histogram_features, axis=1)

    return histogram_features


def histogram_generator(img, bins):
    """
    Generates feature for histogram-based model (single view)
    :param img: Image array
    :param bins: number of buns
    :return: histogram feature
    """
    hist = np.histogram(img, bins=bins, density=False)
    hist_result = hist[0] / (hist[0].sum())
    return hist_result


def load_images(image_path, view):
    """
    Function that loads and preprocess input images
    :param image_path: base path to image
    :param view: L-CC / R-CC / L-MLO / R-MLO
    :return: Batch x Height x Width x Channels array
    """
    image = misc.imread(image_path + view + '.png')
    image = image.astype(np.float32)
    normalize_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image


def normalize_single_image(image):
    """
    Normalize image in-place
    :param image: numpy array
    """
    print np.amax(image)
    image = image.astype('float32')
    image -= np.mean(image)
    image /= np.std(image)
    #image /= np.amax(image)# np.std(image)
    

    
    print np.amax(image)
    
    
def segment_breast(img, low_int_threshold=.05, crop=False):
        '''Perform breast segmentation
        Args:
            low_int_threshold([float or int]): Low intensity threshold to 
                    filter out background. It can be a fraction of the max 
                    intensity value or an integer intensity value.
            crop ([bool]): Whether or not to crop the image.
        Returns:
            An image of the segmented breast.
        NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
            which has a max value of 255.
        '''
        # Create img for thresholding and contours.
        img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
        if low_int_threshold < 1.:
            low_th = int(img_8u.max()*low_int_threshold)
        else:
            low_th = int(low_int_threshold)
        _, img_bin = cv2.threshold(
            img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            contours, _ = cv2.findContours(
                img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(
                img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
        breast_mask = cv2.drawContours(
            np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
        # segment the breast.
        img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
        x, y, w, h = cv2.boundingRect(contours[idx])
        if crop:
            img_breast_only = img_breast_only[y:y+h, x:x+w]
        return img_breast_only, (x,y,w,h)
    
    
def load_dcm_images(image_path):
    """
    Function that loads and preprocess input images
    :param image_path: base path to image
    :param view: L-CC / R-CC / L-MLO / R-MLO
    :return: Batch x Height x Width x Channels array
    """
    image = pyd.dcmread(image_path).pixel_array
    #image = segment_breast(image)[0]

    image = cv2.resize(image, (2000, 2600), interpolation=cv2.INTER_AREA)

    

    #image = image.astype(np.float32)
    normalize_single_image(image)
    print 'normilized', np.amax(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image    

