import cv2
import numpy as np
####
path1 = 'images/test1.jpeg'
path2 = 'path to iamge'

####


img1 = cv2.imread(path1)
img2 = cv2.imread(path2)



# core idea: difference between two gaussian is close to normalized lapacian gaussina

def getGaussianKernel(diameter, sigma):
    if diameter%2 == 0:
        raise ValueError("Diameter must be odd numbmer")
        # Calculate the range for the kernel
    
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # apply normalization
    kernel = kernel / np.sum(kernel)
    return kernel



def siftFeatures(img, scales):
    # create scale space with different gaussian blur
    kernels = []
    for i in range(scales):
        
        pass






    return







image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image_rgb.shape)




