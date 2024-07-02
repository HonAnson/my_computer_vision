import cv2
import numpy as np

# TODO:
# 1. Move nearest circle to click
# 2. Allow 2 images, 8 points to be selected
# 3. display homography of the two images overlapping, with one red and one blue
# 4. Sort the points so that points can be clicked in different order
# 5. Realtime update transformed image so that it represents the homography transformation



        
def updatePointPos(event, x, y, flags, params):
    global circles, img1_resized, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = np.array([x,y])  # move point closest to click to click otherwise
        diff = circles - pos
        dist = diff[:,0]**2 + diff[:,1]**2
        idx = np.argmin(dist)
        circles[idx,:] = [x,y]      



if __name__ == "__main__":
    ### output height and width
    output_width = 600
    output_height = 400
    ### load points 
    path1 = r'images/card1.jpeg'
    img1 = cv2.imread(path1)
    img1_resized = cv2.resize(img1, (0,0), fx = 0.5, fy = 0.5)
    
    # initialize points position
    height, width, _ = img1_resized.shape
    circles = np.array([[width//3, height//3],
                        [width//3, height*2//3],
                        [width*2//3, height//3],
                        [width*2//3, height*2//3]])

    targets = np.array([[0,0],[0,output_height],[output_width,0],[output_width,output_height]])
    
    
    # initialize callback reader
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', updatePointPos)

    while True:
        mask = img1_resized.copy()
        for i in range(4):
            cv2.circle(mask, (circles[i,0], circles[i,1]), 5, (0,255,0), cv2.FILLED )
        cv2.imshow('image', mask)

        # show wrapped image
        homo = cv2.getPerspectiveTransform(np.float32(circles), np.float32(targets))
        img_output = cv2.warpPerspective(img1_resized, homo, (output_width, output_height))

        cv2.imshow('output', img_output)

        if cv2.waitKey(20) & 0xFF == 113:
            break



