# this sift algorithm I implement is gave reference from the paper Anatomy of the SIFT method, by Ives Rey-Otero and Mauricio Delbracio
import cv2
from cv2 import KeyPoint
import numpy as np
from numpy import cos, sin, deg2rad, roll, logical_and, where, inner, exp, rad2deg, arctan2, trace, dot, convolve, sqrt, subtract, log, floor, stack, delete, concatenate, max
from numpy.linalg import det, lstsq, norm
from einops import rearrange
from functools import cmp_to_key

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
    neighbours = concatenate((temp1_unrolled, temp2_unrolled, temp3_unrolled))

    # calculate max and min of neighbours
    neighbours_max = np.max(neighbours)
    neighbours_min = np.max(neighbours)
    
    if centre_pixel_value > neighbours_max or centre_pixel_value < neighbours_min:
        return True
    return False


def getGradient(pixel_cube):
    """ Approximate gradient with central difference formula, error has O(h^2), where h is step size
    """
    dx = 0.5 * (pixel_cube[1,1,2] - pixel_cube[1,1,0])
    dy = 0.5 * (pixel_cube[1,2,1] - pixel_cube[1,0,1])
    dz = 0.5 * (pixel_cube[2,1,1] - pixel_cube[0,1,1])
    return np.array([dx, dy, dz])

def getHessian(pixel_cube):
    """ Approximate hessian with central difference formula, refer material for implementation
    """
    dxx = pixel_cube[1,1,2] - 2*pixel_cube[1,1,1] + pixel_cube[1,1,0]
    dyy = pixel_cube[1,2,1] - 2*pixel_cube[1,1,1] + pixel_cube[1,0,1]
    dss = pixel_cube[2,1,1] - 2*pixel_cube[1,1,1] + pixel_cube[0,1,1]
    dxy = 0.25 * (pixel_cube[1,2,2] - pixel_cube[1,2,0] - pixel_cube[1,0,2] +pixel_cube[1,0,0])
    dxs = 0.25 * (pixel_cube[2,1,2] - pixel_cube[2,1,0] - pixel_cube[0,1,2] +pixel_cube[0,1,0])
    dys = 0.25 * (pixel_cube[2,2,1] - pixel_cube[2,0,1] - pixel_cube[0,2,1] +pixel_cube[0,0,1])

    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys], 
                     [dxs, dys, dss]])



def localizeExtremumByQuadraticFit( i, j, img_idx, octave_idx, 
                                    num_scales, dog_image_in_octave, 
                                    sigma, contrast_threshold, boarder_width, 
                                    eigenvalue_ratio = 10, num_attempts = 5):
    """ Refine pixel position of scale-space extrema with quadratic fit
    """
    is_outside_image = False
    image_shape = dog_image_in_octave[0].shape
    # print(dog_image_in_octave[img_idx-1:img_idx+2])
    for attempt_idx in range(num_attempts):
        img1, img2, img3 = dog_image_in_octave[img_idx-1:img_idx+2]
        pixel_cube = stack([img1[i-1:i+2, j-1:j+2], img2[i-1:i+2, j-1:j+2], img3[i-1:i+2, j-1:j+2]])
        gradient = getGradient(pixel_cube)
        hessian = getHessian(pixel_cube)
        extremum_update = -lstsq(hessian, gradient)[0]
        
        if max(abs(extremum_update)) < 0.5:
            break
        i += int(round(extremum_update[0]))
        j += int(round(extremum_update[1]))
        img_idx += int(round(extremum_update[2]))

        # break conditions
        if i < boarder_width or i >= image_shape[0] - boarder_width or j < boarder_width or j >= image_shape[1] - boarder_width or img_idx > num_scales or img_idx < 1:
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
            keypoint.octave = octave_idx + img_idx * (2**8) + int(round((extremum_update[2]+0.5) * 255)) * (2**16)
            keypoint.size = sigma * (2** ((img_idx + extremum_update[2]) / np.float32(num_scales))) * (2 ** (octave_idx + 1))
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, img_idx
    return None


def getKeypointsWithOrientations(keypoint, octave_idx, gaussian_image, radius_factor = 3, num_bins = 36, peak_ratio = 0.8, scale_factor=1.5):
    """Computer orientation of keypoint
    """
    keypoints_with_orientation = []
    image_shape = gaussian_image.shape
    scale = scale_factor * keypoint.size / np.float32(2**(octave_idx + 1))
    radius = int(round(radius_factor*scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    # we uses square neighbours
    for i in range(-radius, radius+1):    #perhaps can change it to somethign like range(max(0, -radius), min(radius+1, somethign))
        region_y = int(round(keypoint.pt[1] / np.float32(2**octave_idx) + i))
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius+1):
                region_x = int(round(keypoint.pt[0] / np.float32(2**octave_idx) + i))
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x+1] - gaussian_image[region_y, region_x-1]
                    dy = gaussian_image[region_y-1, region_x] - gaussian_image[region_y+1, region_x]
                    grad_magnitude = sqrt(dx**2+dy**2)
                    grad_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i**2 + j**2))         # if i and j are closer to the centre, weight is higher
                    histogram_idx = int(round(grad_orientation * num_bins / 360.))
                    raw_histogram[histogram_idx % num_bins] += weight * grad_magnitude

    smooth_weights = np.array([1/16,1/4,6/16,1/4,1/16])
    for n in range(num_bins):
        arr = np.array([raw_histogram[n-2], raw_histogram[n-1], raw_histogram[n], raw_histogram[(n+1)%num_bins], raw_histogram[(n+2)%num_bins]])
        smooth_histogram[n] = inner(arr, smooth_weights)

    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram < roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index-1) % num_bins]
            right_value = smooth_histogram[(peak_index+1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2*peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 10e-5:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientation.append(new_keypoint)
    
    return keypoints_with_orientation

def getScaleSpaceExtrema(gaussian_images, dog_images, num_scales, sigma, boarder_width = 5, contrast_threshold = 0.4):
    """Find pixel position in all scale spaces that are Extrema
    """
    threshold = floor(0.5 * contrast_threshold / num_scales * 255) # from reference material
    keypoints = []
    count = 0
    for octave_idx, dog_images_in_octave in enumerate(dog_images):
        for img_idx, (img1, img2, img3) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            i_bound = img1.shape[0] - boarder_width
            j_bound = img1.shape[1] - boarder_width
            for i in range(boarder_width, i_bound):
                count += 1
                if count % 100 == 0:
                    print(count)
                for j in range(boarder_width, j_bound):
                    is_extremum = isExtremum(img1[i-1:i+2, j-1:j+2], img2[i-1:i+2, j-1:j+2], img3[i-1:i+2, j-1:j+2], contrast_threshold)
                    if is_extremum:
                        localization_result = localizeExtremumByQuadraticFit(i, j, img_idx + 1, octave_idx, num_scales, dog_images_in_octave, sigma, contrast_threshold, boarder_width) 
                        if localization_result is not None:
                            keypoint, localized_image_idx = localization_result
                            keypoints_with_orientations = getKeypointsWithOrientations(keypoint, octave_idx, gaussian_images[octave_idx][localized_image_idx])
                            keypoints.append(keypoints_with_orientations)

    return keypoints


def compareKeyPoints(keypoint1, keypoint2):
    """ Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicates
    """

    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key = cmp_to_key(compareKeyPoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if  last_unique_keypoint.pt[0] != next_keypoint.pt[0] or\
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or\
            last_unique_keypoint.size != next_keypoint.size or\
            last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    
    return unique_keypoints


def convertKeyPointsToImageSize(keypoints):
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5*np.array(keypoint.pt))
        keypoint.size*= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1)& 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints


### Generate Descriptors ###
def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128  
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def getDescriptors(keypoints, gaussian_image, window_width = 4, num_bins=8, scale_multipler=3, descriptor_max_value=0.2):
    """Generate descriptor for each keypoint
    """

    descriptors = []
    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave+1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))


    return


def sift(image, sigma = 1, num_scales = 3, blur = 0.5, img_border_width = 5):
    num_octave = int(round(log(min(image.shape))/log(2) - 1)) # number of times we can half the image before it is too small
    kernel_sizes = getKernelSizes(sigma, num_scales)    
    gaussian_images = getGaussianImages(image, num_octave, kernel_sizes)
    DoG_images = getDoG(gaussian_images)
    keypoints = getScaleSpaceExtrema(gaussian_images, DoG_images, num_scales, 5)



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

    ### Parameters ###
    num_scale = 3
    boarder_width = 5
    sigma = 1 


    num_octave = int(floor(log(min(img1_gray.shape))/log(2) - 3)) # number of times we can half the image before it is too small
    kernel_sizes = getKernelSizes(1, 3)
    gaussian_images = getGaussianImages(img1_gray, num_octave, kernel_sizes)
    DoG_images = getDoG(gaussian_images)
    keypoints = getScaleSpaceExtrema(gaussian_images, DoG_images, num_scale, sigma)
    
    # output = DoG_images[2][0]
    # ##### Display image ####
    # fx = 1
    # fy = 1
    # resized_image = cv2.resize(output, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    # cv2.imshow('Grayscale Image', resized_image)
    # while True:
    #     if cv2.getWindowProperty('Grayscale Image', cv2.WND_PROP_VISIBLE) < 1:  
    #         break
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
    #         break
    # cv2.destroyAllWindows()  # Close the image window



