import cv2
import numpy as np

# TODO:
# 1. Move nearest circle to click
# 2. Allow 2 images, 8 points to be selected
# 3. display homography of the two images overlapping, with one red and one blue
# 4. Sort the points so that points can be clicked in different order
# 5. Realtime update transformed image so that it represents the homography transformation



        
def updatePointPos(event, x, y, flags, params):
    global counter, circles, img1_resized, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = np.array([x,y])  # move point closest to click to click otherwise
        diff = circles - pos
        dist = diff[:,0]**2 + diff[:,1]**2
        idx = np.argmin(dist)
        circles[idx,:] = [x,y]      



if __name__ == "__main__":
    circles = np.zeros((4,2), int)
    counter = 0
    path1 = r'images/london1.jpeg'
    path2 = r'images/card2.jpeg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1_resized = cv2.resize(img1, (0,0), fx = 0.5, fy = 0.5)
    img2_resized = cv2.resize(img2, (0,0), fx = 0.5, fy = 0.5)
    
    # initialize points position
    height, width, _ = img1_resized.shape
    circles = np.array([[width//3, height//3],
                        [width//3, height*2//3],
                        [width*2//3, height//3],
                        [width*2//3, height*2//3]])

    # output height and width    
    output_width = 400
    output_height = 400
    targets = np.array([[0,0],[0,height],[width,0],[width,height]])
    
    
    # initialize callback reader
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', updatePointPos)

    while True:
        mask = img1_resized.copy()
        for i in range(4):
            cv2.circle(mask, (circles[i,0], circles[i,1]), 5, (0,255,0), cv2.FILLED )
        cv2.imshow('image', mask)

        # show wrapped image
        homo = cv2.getPerspectiveTransform(circles, targets)
        # img_output = cv2.warpPerspective(img1_resized, homo, (output_width, output_height))





        if cv2.waitKey(20) & 0xFF == 113:
            break



