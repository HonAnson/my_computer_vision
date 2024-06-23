import cv2
import numpy as np
from sift import sift



def matchKeyPoints(keypoints1, keypoints2, descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def getSteroTransformation(corrspondance):



    return

def randsac(matches):

    return

def stero(image1, image2):

    
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

