import cv2
import numpy as np
from sift import sift, gaussianSmoothing



####
path1 = 'images/test1.jpeg'
path2 = 'images/test2.jpeg'
####


img1 = cv2.imread(path1)
img2 = cv2.imread(path2)


# core idea: difference between two gaussian is close to normalized lapacian gaussina


if __name__ == '__main__':
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # image_gray = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)



    ### Display image
    output = gaussianSmoothing(img1_gray, 20)
    fx = 0.4
    fy = 0.4
    resized_image = cv2.resize(output, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    cv2.imshow('Grayscale Image', resized_image)

    while True:
        if cv2.getWindowProperty('Grayscale Image', cv2.WND_PROP_VISIBLE) < 1:  
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
    cv2.destroyAllWindows()  # Close the image window
    print(output.shape)

