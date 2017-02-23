import numpy as np
from matplotlib import pyplot as plt
import cv2
import os


def loadImage(src):
    img = cv2.imread(src, 1)
    return img


def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO: When giving an image full of blue pixels, it's showing a black image. Find why!


def isSkin(img):
    """Return an image of white pixels """
    res = np.array(
        [np.array(
            [np.array(
                [0 for k in range(3)], dtype=np.uint8)
             for j in range(len(img[0]))], dtype=np.uint8)
         for i in range(len(img))], dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            res[i][j] = rgbSkinPixel(img[i][j])
    return res


def rgbSkinPixel(pixel):
    """Return a white pixel if the input is a skin pixel"""
    B, G, R = pixel[0], pixel[1], pixel[2]
    if R > 95 and G > 40 and R > 20 and\
        max(pixel) - min(pixel) > 15 and\
        abs(int(R) - int(G)) > 15 and\
        R > G and R > B:
        return np.array([255, 255, 255], dtype=np.uint8)
    return np.array([0, 0, 0], dtype=np.uint8)


if __name__ == '__main__':
    dataset_path = 'C:/Users/L Palmito/PyCharmProjects/visord/tp1/Pratheepan_Dataset/FacePhoto/'
    ground_truth_path = 'C:/Users/L Palmito/PyCharmProjects/visord/tp1/Ground_Truth/GroundT_FacePhoto/'

    for image_name in os.listdir(dataset_path)[:2]:
        img = loadImage(dataset_path + image_name)
        skin_map = isSkin(img)
        compared_images = np.concatenate((img, skin_map), axis=1)
        showImage(compared_images)

        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # # lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        #
        # hsv_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # plt.imshow(hsv_hist, interpolation='nearest')
        # plt.show()

    # lab_hist = cv2.calcHist([lab], [1, 2], None, [180, 256], [0, 180, 0, 256])
