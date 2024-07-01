import cv2
import numpy as np
# from sift import sift
from random import shuffle
from svd import svd
from einops import rearrange



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

def det(M):
    """Return determinate of a 3x3 matrix"""
    first = M[0,0]*(M[1,1]*M[2,2]-M[1,2]*M[2,1])
    second = M[0,1]*(M[1,0]*M[2,2]-M[1,2]*M[2,0]) 
    third = M[0,2]*(M[1,0]*M[2,1]-M[1,1]*M[2,0])
    return first - second + third


def EstimateFundamentalMatrixRandsac(points1, points2, num_trials, threshold):
    """ Estimate fundamental matrix with randsac using given corrspondance pairs"""
    best_inlier = 0
    best_trans = np.zeros((3,3))
    idx = np.arange(len(points1))
    for _ in range(num_trials):
        np.random.shuffle(idx)

        # rewriting constraint equation from v'*f_m*v = 0 to A*f_v = 0, where f_m is fundamental matrix
        # and f_v is the unrolled fundamental matrix
        A = np.zeros((8,9))
        for i in range(8):
            u1, v1 = points1[idx[i],:]
            u2, v2 = points2[idx[i],:]
            A[i,:] = np.array([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1])  # can be worked out by expanding constraint equatin

        # Solve for non-trival nullspace of matrix A
        _, _, V = svd(A)
        f_v = V[:,-1]               # choosing last column vector of V
        f_m = f_v.reshape((3,3))    # NOTE: rank of f_m is already 2
        f_m = f_m / det(f_m)

        # convert points into homogenous coordinates
        eval_1, eval_2 = np.ones((len(points1),3)), np.ones((points1,3))
        eval_1[:,0:2] = points1
        eval_2[:,0:2] = points2
        total = np.sum(((eval_1 @ f_m) * eval_2), axis = 1)
        # complete randsac by checking inliers
        inliers_count = (total < threshold).sum()
        if inliers_count > best_inlier:
            best_inlier = inliers_count
            best_trans = f_m
            mask = total < threshold
    return best_trans, mask



def compute_rectification_homographies(F, img1, img2, pts1, pts2):
    """Compute the rectification homographies."""
    h, w = img1.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))
    return H1, H2

def apply_homographies(img1, img2, H1, H2):
    """Apply the rectifying homographies to the images."""
    h, w = img1.shape
    rectified_img1 = cv2.warpPerspective(img1, H1, (w, h))
    rectified_img2 = cv2.warpPerspective(img2, H2, (w, h))
    return rectified_img1, rectified_img2



def stero(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(image2, None)
    matches = matchKeyPoints(descriptor1, descriptor2)
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.queryIdx].pt for m in matches])



    fundamental_matrix, mask = EstimateFundamentalMatrixRandsac(points1, points2, 100, 50)



    return fundamental_matrix, mask
    

if __name__ == '__main__':
    ####
    path1 = 'images/card1.jpeg'
    path2 = 'images/card2.jpeg'
    ####

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    f = stero(img1_gray, img2_gray)

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

