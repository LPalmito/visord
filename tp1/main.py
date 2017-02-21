import numpy as np
from matplotlib import pyplot as plt
import cv2


def loadImage(src):
    img = cv2.imread(src, 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


# TODO: When giving an image full of blue pixels, it's showing a black image. Find why!


def isSkin(img):
    """Return an image of white pixels """
    res = np.array([np.array([np.array([0 for k in range(3)]) for j in range(len(img[0]))]) for i in range(len(img))])
    for i in range(len(img)):
        for j in range(len(img[0])):
            res[i][j] = rgbSkinPixel(img[i][j])
    return res


def rgbSkinPixel(pixel):
    """Return a white pixel if the input is a skin pixel"""
    # B, G, R = pixel[0], pixel[1], pixel[2]
    # if R > 95 and G > 40 and R > 20 and\
    #     max(pixel) - min(pixel) > 15 and\
    #     abs(int(R) - int(G)) > 15 and\
    #     R > G and R > B:
    #     return np.array([255, 255, 255])
    # return np.array([0, 0, 0])
    return np.array([255, 0, 0])


if __name__ == '__main__':
    path = 'C:/Users/L Palmito/PyCharmProjects/visord'
    img = loadImage(path + '/tp1/Casting.jpg')
    res = isSkin(img)
    cv2.imshow('skin detector', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
