# this sift algorithm I implement is gave reference from the paper Anatomy of the SIFT method, by Ives Rey-Otero and Mauricio Delbracio
import cv2
from cv2 import KeyPoint
import numpy as np
from numpy import trace, dot, convolve, sqrt, subtract, log, floor, stack, delete, concatenate, max
from numpy.linalg import det, lstsq, norm
from einops import rearrange


def gaussianBlurring(image, sigma):
    """ Apply gaussian blurring to given image
    """
    bound = np.ceil(4*sigma)
    low_b, up_b = -bound, bound
    k = np.arange(low_b, up_b+1, 1)
    # get gaussian values
    g = np.exp(-(np.square(k)) / (2*sigma**2))
    g /= sum(g)

    # take convolution
    canvas = np.zeros_like(image)
    for i in range(image.shape[0]):
        canvas[i,:] = convolve(g, image[i,:], mode='same')
    for j in range(image.shape[1]):
        canvas[:,j] = convolve(g, canvas[:,j], mode='same')
   
    return canvas




def getKernelSizes(sigma, scales):
    ### generate a list of kernel size we need
    num_img_per_oct = scales + 3 #as per Ives and Mauricio's paper
    k = 2** (1. / scales) # after 1 octive, sigma doubles
    gaussian_kernels = np.zeros(num_img_per_oct)
    gaussian_kernels[0] = sigma
    
    for scale_idx in range(1, num_img_per_oct):
        sigma_previous = (k** (scale_idx - 1))*sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[scale_idx] = sqrt(sigma_total**2 - sigma_previous**2)  #recall pythogerian identity for gaussian convolution
    return gaussian_kernels



def getGaussianImages(image, num_octaves, kernels):
    ### Generate blurred image in each octave
    gaussian_images = []
    for octave_idx in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)
        for kernel_size in kernels[1:]:
            image = gaussianBlurring(image, kernel_size)
            gaussian_images_in_octave.append(image)
        
        gaussian_images.append(np.array(gaussian_images_in_octave))
        octave_base = gaussian_images_in_octave[-3]
        fx = 0.5
        fy = 0.5
        image = cv2.resize(octave_base, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    return gaussian_images



def getDoG(gaussian_images):
    """ Return difference of gaussian images from list of gaussian images in different octave
    """
    dog_images = []
    for gaussian_iamges_in_octave in gaussian_images:
        dog_image_in_octave = []
        for less_blurred_image, more_blurred_image in zip(gaussian_iamges_in_octave, gaussian_iamges_in_octave[1:]):
            dog_image_in_octave.append(subtract(more_blurred_image, less_blurred_image))
        dog_images.append(np.array(dog_image_in_octave))
    return dog_images



def isExtremum(temp1, temp2, temp3, threshold):
    """ Checking a paxiel is a extremum among its neighbors, return boolean value of whether this is true
    """
    centre_pixel_value = temp2[1,1]
    if abs(centre_pixel_value) < threshold or centre_pixel_value == 0:
        return False
    # get neighbours
    temp1_unrolled = rearrange(temp1, 'm n -> (m n)')
    temp2_unrolled = rearrange(temp2, 'm n -> (m n)')
    temp3_unrolled = rearrange(temp3, 'm n -> (m n)')
    temp2_unrolled = delete(temp2, 4)       # delete the centre pixel
    neighbours = concatenate(temp1_unrolled, temp2_unrolled, temp3_unrolled)

    # calculate max and min of neighbours
    neighbours_max = np.max(neighbours)
    neighbours_min = np.max(neighbours)
    
    if centre_pixel_value > neighbours_max or centre_pixel_value < neighbours_min:
        return True
    return False


def getGradient(pixel_cube):
    return

def getHessian(pixel_cube):
    return



def localizeExtremumByQuadraticFit( i, j, img_idx, octave_idx, 
                                    num_scales, dog_image_in_octave, 
                                    sigma, contrast_threshold, boarder_width, 
                                    eigenvalue_ratio = 10, num_attempts = 5):
    """ Refine pixel position of scale-space extrema with quadratic fit
    """
    is_outside_image = False
    image_shape = dog_image_in_octave[0].shape
    for attempt_idx in range(num_attempts):
        img1, img2, img3 = dog_image_in_octave[img_idx]
        pixel_cube = stack([img1[i-1:i+2], img2[i-1:i+2], img3[i-1:i+2]])
        gradient = getGradient(pixel_cube)
        hessian = getHessian(pixel_cube)
        extremum_update = -lstsq(hessian, gradient)[0]
        
        if max(abs(extremum_update)) < 0.5:
            break
        i += int(round(extremum_update[0]))
        j += int(round(extremum_update[1]))
        img_idx += int(round(extremum_update[2]))

        # break conditions
        if i < boarder_width or i >= image_shape[0] - boarder_width or j < boarder_width or j > image_shape[1] - boarder_width:
            is_outside_image = True
            break
        
        if is_outside_image:
            return None
        
        if attempt_idx > num_attempts:
            return None
        
        functionValueAtUpdatedExtremum = pixel_cube[1,1,1] + 0.5 * dot(gradient, extremum_update)   #learning rate of 0.5
        if abs(functionValueAtUpdatedExtremum) * num_scales > contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace**2) < ((eigenvalue_ratio + 1)**2)* xy_hessian_det:
                # Contrast check passed -- construct and return opencv KeyPoint object
                keypoint = KeyPoint()
                scale = 2**octave_idx
                keypoint.pt = ((j + extremum_update[0]) * scale, (i + extremum_update[1]) * scale)
                keypoint.octave = octave_idx + img_idx * (2**8) / float32(num_e)
e









    return

def getScaleSpaceExtrema(gaussian_images, dog_images, num_scales, sigma, boarder_width, contrast_threshold = 0.4):
    """Find pixel position in all scale spaces that are Extrema
    """
    threshold = floor(0.5 * contrast_threshold / num_scales * 255) # from reference material

    for octave_idx, dog_images_in_octave in enumerate(dog_images):
        for img_idx, (img1, img2, img3) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            i_bound = img1.shape[0] - boarder_width
            j_bound = img1.shape[1] - boarder_width
            for i in range(boarder_width, i_bound):
                for j in range(boarder_width, j_bound):
                    is_extremum = isExtremum(img1[i-1:i+2], img2[i-1:i+2], img3[i-1:i+2])
                    if is_extremum:
                        localization_result = localizeExtremumByQuadraticFit(i, j, img_idx + 1, octave_idx, num_scales, dog_images_in_octave, sigma, contrast_threshold, boarder_width) 



    return



def sift(image, sigma = 1, scales = 3, blur = 0.5, img_border_width = 5):
    num_octave = int(round(log(min(image.shape))/log(2) - 1)) # number of times we can half the image before it is too small
    kernel_sizes = getKernelSizes(sigma, scales)    
    gaussian_images = getGaussianImages(image, num_octave, kernel_sizes)
    DoG_images = getDoG(gaussian_images)


    return 



if __name__ == '__main__':

    ####
    path1 = 'images/test1.jpeg'
    path2 = 'images/test2.jpeg'
    ####

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # image_gray = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)

    ### Display image
    num_octave = int(floor(log(min(img1_gray.shape))/log(2) - 3)) # number of times we can half the image before it is too small
    kernel_sizes = getKernelSizes(1, 3)
    gaussian_images = getGaussianImages(img1_gray, num_octave, kernel_sizes)
    DoG_images = getDoG(gaussian_images)
    print(DoG_images[0].shape)
    
    output = DoG_images[2][0]
    ##### Display image ####
    fx = 1
    fy = 1
    resized_image = cv2.resize(output, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    cv2.imshow('Grayscale Image', resized_image)
    while True:
        if cv2.getWindowProperty('Grayscale Image', cv2.WND_PROP_VISIBLE) < 1:  
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    cv2.destroyAllWindows()  # Close the image window



