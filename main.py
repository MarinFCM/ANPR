import os
import cv2
import numpy as np

from keras.layers import *
from keras.models import Sequential
from keras.losses import *
from keras.optimizers import *
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocessImg(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    resize_height = 50
    resize_width = 20
    #image = cv2.resize(image, (resize_height, resize_width))
    #print(image.shape)
    image = tf.image.resize(rgb_tensor, (resize_height, resize_width))
    image /= 255.0
    image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

    return image


def pack_features_vectorImg(features):
    features = tf.reshape(features, (50, 20, 1))
    return features


directory = r'C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeCrop'
results = []
name = 1

new_model = tf.keras.models.load_model('my_model')
#new_model.summary()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']

for filename in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, filename))
    height, width, ch = img.shape
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (360, 80))
    imgT = cv2.warpPerspective(img, M, (360, 80))

    hsv = cv2.cvtColor(imgT, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([0, 0, 100])
    brown_hi = np.array([255, 255, 255])

    lower_blue = np.array([60, 60, 0])
    upper_blue = np.array([215, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    imgT[mask > 0] = (255, 255, 255)

    plate_image = cv2.convertScaleAbs(dst, alpha=255.0)

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
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
    alpha = 1.8
    beta = 25.0
    # binarize
    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    new_image = np.zeros(imgT.shape, imgT.dtype)
    for y in range(imgT.shape[0]):
        for x in range(imgT.shape[1]):
            for c in range(imgT.shape[2]):
                new_image[y, x, c] = np.clip(alpha * imgT[y, x, c] + beta, 0, 255)

    result = new_image.copy()
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
    rois = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 2 * w or h < 25 or w < 5:
            continue
        # Getting ROI
        roi = dst[y:y + h, x:x + w]
        rois.append(preprocessImg(roi))
        cv2.rectangle(dst, (x, y), (x + w, y + h), (90, 0, 255), 2)
        #ev = new_model.evaluate(pack_features_vectorImg(preprocessImg(img)))
        #print(ev)
        #preds.append(class_names[pred])
        # show ROI
        #cv2.imwrite('C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeTest/' + filename[part] + '_' + str(name) + '.png', roi)
    if len(rois) == 0:
        print("FILENAME: ", filename, " FAILED")
        continue
    df = tf.data.Dataset.from_tensor_slices(rois)
    ev = new_model.predict(df.map(pack_features_vectorImg).batch(1))
    preds = ev.argmax(axis=1)
    output = []
    for p in preds:
        output.append(class_names[p])
    print("FILENAME: ", filename)
    print("PREDICTION: ", output)
    break
    # cv2.imshow('contrast', result)
    # cv2.imshow('marked areas', dst)
    # cv2.waitKey(0)
    #results.append(dst)
