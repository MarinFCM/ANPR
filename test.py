import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

directory = r'C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeCrop'
results = []
name = 1
for filename in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, filename))
    height, width, ch = img.shape
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (360, 80))
    imgAvgBrightness = brightness(dst)
    new_image = np.zeros(dst.shape, dst.dtype)
    if imgAvgBrightness < 150:
        alpha = 1.6
        beta = 20.0
        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                for c in range(dst.shape[2]):
                    new_image[y, x, c] = np.clip(alpha * dst[y, x, c] + beta, 0, 255)

    if imgAvgBrightness < 150:
        hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
    else:
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([0, 0, 100])
    brown_hi = np.array([255, 255, 255])

    lower_blue = np.array([60, 60, 100])
    upper_blue = np.array([215, 255, 255])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Change image to red where we found brown
    dst[mask > 0] = (255, 255, 255)
    height, width, ch = img.shape

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Applied inversed thresh_binary
    binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    (h, w) = binary.shape
    a = [0 for z in range(0, w)]
    # print(a)

    for j in range(0, w):
        for i in range(0, h):
            if binary[i, j] == 0:
                a[j] += 1
                binary[i, j] = 255

    for j in range(0, w):
        for i in range((h - a[j]), h):
            binary[i, j] = 0

    thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    (h, w) = thresh1.shape  # Return height and width
    a = [0 for z in range(0, h)]

    for j in range(0, h):
        for i in range(0, w):
            if thresh1[j, i] == 0:
                a[j] += 1
                thresh1[j, i] = 255
    for j in range(0, h):
        for i in range(0, a[j]):
            thresh1[j, i] = 0

    xStart = []
    xStop = []
    flow = False
    for i in range(0, w):
        if binary[7][i] == 255 and flow is False:
            flow = True
            xStart.append(i)
        if binary[7][i] == 0 and flow is True:
            flow = False
            xStop.append(i)

    yStart = []
    yStop = []
    flow = False
    for i in range(2, h):
        if thresh1[i][310] == 255 and flow is False:
            flow = True
            yStart.append(i)
        if thresh1[i][310] == 0 and flow is True:
            flow = False
            yStop.append(i)

    if len(yStart) > len(yStop):
        yStop.append(h)
        #results.append(new_image)
    thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if imgAvgBrightness < 150:
        result = new_image.copy()
    else:
        result = dst.copy()
    result[thresh == 0] = (255, 255, 255)
    gray1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray1)
    blur1 = cv2.GaussianBlur(cl1, (7, 7), 0)
    ret, thresh1 = cv2.threshold(blur1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find contours
    ctrs, hier = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    part = 0
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 2 * w or h < 25 or w < 5:
            continue
        # Getting ROI
        roi = dst[y:y + h, x:x + w]

        # show ROI
        #cv2.imwrite('C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeTest/' + filename[part] + '_' + str(name) + '.png', roi)
        part += 1
        name = name + 1
        #cv2.imshow('charachter' + str(i), roi)
        cv2.rectangle(dst, (x, y), (x + w, y + h), (90, 0, 255), 2)

    # cv2.imshow('contrast', result)
    # cv2.imshow('marked areas', dst)
    # cv2.waitKey(0)
    results.append(dst)
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
