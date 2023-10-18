import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"
Myimage = cv.imread(url_img, cv.IMREAD_GRAYSCALE)

hist_gray = cv.calcHist([Myimage], [0], None, [256], [0, 256])
plt.figure(figsize=(6, 3))

plt.plot(hist_gray, color='k')
plt.title('Gray Histogram')
plt.xlim([0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
img1 = cv.imread(url_img)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])

original_image = cv.imread(url_img, cv.IMREAD_GRAYSCALE)
dark_image = cv.convertScaleAbs(original_image, alpha=0.5, beta=0)
gray_image = cv.convertScaleAbs(original_image, alpha=1, beta=0)
bright_image = cv.convertScaleAbs(original_image, alpha=1.5, beta=0)
white_image = cv.convertScaleAbs(original_image, alpha=2, beta=100)

images = [dark_image, gray_image, bright_image, white_image]
titles = ['Dark', 'Gray', 'Bright', 'White']

plt.figure(figsize=(12, 4))
for i in range(len(images)):
    plt.subplot(1, 4, i + 1)
    plt.hist(images[i].ravel(), bins=256, range=(0, 256), density=True, color='gray')
    plt.title(titles[i])
    plt.xlim([0, 256])

plt.tight_layout()

plt.show()