import numpy as np
import cv2


def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


imagePath = "C:/Users/Marin/Desktop/6.SEM/Zavrsni/regeCrop/ZG4892GI.jpg"
img = cv2.imread(imagePath)
results = []
height, width, ch = img.shape
pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts2 = np.float32([[0, 0], [360, 0], [0, 80], [360, 80]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (360, 80))
results.append(dst)
imgT = cv2.warpPerspective(img, M, (360, 80))
imgAvgBrightness = brightness(dst)

new_image = np.zeros(imgT.shape, imgT.dtype)
if imgAvgBrightness < 150:
    alpha = 1.6
    beta = 20.0
    for y in range(imgT.shape[0]):
        for x in range(imgT.shape[1]):
            for c in range(imgT.shape[2]):
                new_image[y, x, c] = np.clip(alpha * imgT[y, x, c] + beta, 0, 255)
    results.append(new_image)

gamma=0.4
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv2.LUT(dst, lookUpTable)
#results.append(res)

if imgAvgBrightness < 150:
    hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
else:
    hsv = cv2.cvtColor(imgT, cv2.COLOR_BGR2HSV)

# Define lower and uppper limits of what we call "brown"
brown_lo = np.array([0, 0, 100])
brown_hi = np.array([255, 255, 255])

lower_blue = np.array([60, 60, 100])
upper_blue = np.array([215, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

if imgAvgBrightness < 150:
    new_image[mask > 0] = (255, 255, 255)
else:
    imgT[mask > 0] = (255, 255, 255)
np_vert = np.vstack(results)
cv2.imshow("vertical4", np_vert)
cv2.waitKey()