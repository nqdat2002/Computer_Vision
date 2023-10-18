import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 2D Convolution ( Image Filtering )
url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"
def test2DFilter():
    img = cv.imread(url_img)
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
    img = cv.imread(url_img)
    assert img is not None, "file could not be read, check with os.path.exists()"
    blur = cv.blur(img,(5,5))
    boxFilter = cv.boxFilter(img, -1, (5, 5))
    gausianblur = cv.GaussianBlur(img,(5,5),0)
    median = cv.medianBlur(img,5)
    bilateral = cv.bilateralFilter(img, 9, 75, 75)
    # original
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])

    s = int(input())
    if s == 0:
        plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
    if s == 1:
        plt.subplot(122), plt.imshow(boxFilter), plt.title('BoxFilter')
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

import cv2
import numpy as np
import matplotlib.pyplot as plt
url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"

def CreateNoise():
    # Đọc ảnh gốc
    image = cv2.imread(url_img)

    # Tạo nhiễu Gaussian
    mean = 0
    stddev = 25
    gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image_gaussian = cv2.add(image, gaussian_noise)

    # Tạo nhiễu muối và tiêu
    salt_prob = 0.01
    pepper_prob = 0.01

    salt = (np.random.random(image.shape[:2]) < salt_prob) * 255
    pepper = (np.random.random(image.shape[:2]) < pepper_prob) * 255

    noisy_image_salt_pepper = image.copy()
    noisy_image_salt_pepper[salt == 255] = 255
    noisy_image_salt_pepper[pepper == 255] = 0

    kernel_size = 3
    filtered_gaussian = cv2.GaussianBlur(noisy_image_gaussian, (kernel_size, kernel_size), 0)
    filtered_median = cv2.medianBlur(noisy_image_salt_pepper, kernel_size)
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(noisy_image_gaussian, cv2.COLOR_BGR2RGB))
    plt.title('Nhiễu Gaussian')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(filtered_gaussian, cv2.COLOR_BGR2RGB))
    plt.title('Lọc Gaussian')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(noisy_image_salt_pepper, cv2.COLOR_BGR2RGB))
    plt.title('Nhiễu Muối và Tiêu')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(filtered_median, cv2.COLOR_BGR2RGB))
    plt.title('Lọc Median')
    plt.axis('off')

    plt.show()

CreateNoise()