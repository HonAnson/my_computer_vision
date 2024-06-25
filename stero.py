import cv2
import numpy as np
from sift import sift
from random import shuffle


def matchKeyPoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


def getSteroTransformation(corrspondance):


    return

def randsac(matches, num_trials):
    best_dist = 10000000
    best_trans = np.zeros((3,3))
    for n in num_trials:
        shuffle(matches)

        
        image1_points = np.zeros((8))
        image2_points = np.zeros((8))

        # write position of corrspondance into arrays
        for i in range(4):
            image1_idx = matches[i].queryIdx
            image2_idx = matches[i].trainIdx
            x1, y1 = keypoints1[image1_idx].pt
            x2, y2 = keypoints2[image1_idx].pt
            image1_points[2*i] = x1
            image1_points[2*i+1] = y1
            image2_points[2*i] = x2
            image2_points[2*i+1] = y2



        transformation = getSteroTransformation(selected_pairs)
        transformed_points = matches[0] @ transformation
        mean_dist = getMeanDistance(transformed_points, matches[1])
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_trans = transformation
    
    return best_trans



def stero(image1, image2):
    keypoints1, descriptor1 = sift(image1)
    keypoints2, descriptor2 = sift(image2)
    correspondance = matchKeyPoints(descriptor1, descriptor2)
    transformation = randsac(correspondance, 100)
    
    
    return
    




if __name__ == '__main__':
    ####
    path1 = 'images/test1.jpeg'
    path2 = 'images/test2.jpeg'
    ####

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


    ### Display image
    fx = 0.4
    fy = 0.4
    resized_image = cv2.resize(img1_gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    cv2.imshow('Grayscale Image', resized_image)

    while True:
        if cv2.getWindowProperty('Grayscale Image', cv2.WND_PROP_VISIBLE) < 1:  
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    cv2.destroyAllWindows()  # Close the image window

