import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 2D Convolution ( Image Filtering )

def test2DFilter():
    img = cv.imread('img1.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv.filter2D(img, -1, kernel)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()


# Image Blurring (Image Smoothing): Averaging, Gaussian, Median, Bilateral Filtering

def testBlurring():
    img = cv.imread('img1.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    blur = cv.blur(img,(5,5))
    gausianblur = cv.GaussianBlur(img,(5,5),0)
    median = cv.medianBlur(img,5)
    bilateral = cv.bilateralFilter(img, 9, 75, 75)
    # original
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])

    s = int(input())
    if s == 1:
        plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
    if s == 2:
        plt.subplot(122), plt.imshow(gausianblur), plt.title('Gausianblur')
        plt.xticks([]), plt.yticks([])
    if s == 3:
        plt.subplot(122), plt.imshow(median), plt.title('Median')
        plt.xticks([]), plt.yticks([])
    if s == 4:
        plt.subplot(122), plt.imshow(bilateral), plt.title('billteral')
        plt.xticks([]), plt.yticks([])
    if s == 5:
        return
    plt.show()


testBlurring()