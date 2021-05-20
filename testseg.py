import cv2
import numpy as np
import os

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([60,60,0])
    upper_blue = np.array([215,255,120])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask


# Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
segmentation_spacing = 0.9
directory = r'C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeCrop'
results = []
name = 1
for filename in os.listdir(directory):

    img = cv2.imread(os.path.join(directory, filename))

    height, width, ch = img.shape
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (360, 80))
    height, width, ch = img.shape

    imgTest = img
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # Define lower and uppper limits of what we call "brown"
    blue_lo=np.array([160,0,40])
    blue_hi=np.array([255,250,255])


    lower_black = np.array([0, 0, 0])
    upper_black = np.array([25, 25, 25])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    #img[mask > 0] = (0, 0, 0)

    lower_blue = np.array([60,60,100])
    upper_blue = np.array([215,215,255])
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    img[mask>0]=(255,255,255)



    gray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, img_threshold = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #ret = cv2.bitwise_not(ret)
    #cv2.imshow('Char', img_threshold)
    #cv2.waitKey()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    '''2 Binary the grayscale image'''
    #ret, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)


    '''3 Split characters'''
    white = []  # Record the sum of white pixels in each column
    black = []  # Record the sum of black pixels in each column
    height = img_threshold.shape[0]
    width = img_threshold.shape[1]

    white_max = 0
    black_max = 0


    '''4 Cycle through the sum of black and white pixels for each column'''
    for i in range(width):
        white_count = 0
        black_count = 0
        for j in range(height):
            if img_threshold[j][i] == 255:
                white_count += 1
            else:
                black_count += 1

        white.append(white_count)
        black.append(black_count)

    white_max = max(white)
    black_max = max(black)


    '''5 Split the image, given the starting point of the character to be split'''
    def find_end(start):
        end = start + 1
        for m in range(start + 1, width - 1):
            if(black[m] > segmentation_spacing * black_max):
                end = m
                break
        return end


    n = 1
    start = 1
    end = 2
    while n < width - 1:
        n += 1
        if(white[n] > (1 - segmentation_spacing) * white_max):
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                #print(start, end)
                character = img_threshold[1:height, start:end]
                cv2.rectangle(img, (start, 1), (end, height), (90, 0, 255), 2)
                #cv2.imshow('Char', cv2.bitwise_not(character))
                #cv2.waitKey()
                #cv2.imwrite('img/{0}.png'.format(n), character)

    results.append(img)
array = np.array_split(results, 5)
np_vert = np.vstack(array[0])
np_vert1 = np.vstack(array[1])
np_vert2 = np.vstack(array[2])
np_vert3 = np.vstack(array[3])
np_vert4 = np.vstack(array[4])
cv2.imshow("vertical", np_vert)
cv2.imshow("vertical1", np_vert1)
cv2.imshow("vertical2", np_vert2)
cv2.imshow("vertical3", np_vert3)
cv2.imshow("vertical4", np_vert4)
cv2.waitKey()