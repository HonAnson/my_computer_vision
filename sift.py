# this sift algorithm I implement is gave reference from the paper Anatomy of the SIFT method, by Ives Rey-Otero and Mauricio Delbracio
import cv2
import numpy as np

def gaussianSmoothing(image, sigma):
    bound = np.ceil(4*sigma)
    low_b, up_b = -bound, bound
    k = np.arange(low_b, up_b, 1)
    # get gaussian values
    g = np.exp(-(np.square(k)) / (2*sigma**2))
    
    return


def getScaleSpace():


    return





def sift(image, sigma = 1.6, num_interval = 3, blur = 0.5, img_border_width = 5):
    
    img = readImage(path)



    
    scale_space = getScaleSpace()


    return keypoints, descriptors













