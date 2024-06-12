import cv2
import numpy as np
####
path1 = 'images/test1.jpeg'
path2 = 'path to iamge'

####


image = cv2.imread(path1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image_rgb.shape)




