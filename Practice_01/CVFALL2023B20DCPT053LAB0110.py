import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"

# Scaling
def Scale():
    img = cv.imread(url_img)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

    height, width = img.shape[:2]
    res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
    cv.imshow('img', res)

# Translation
def Translation():
    img = cv.imread(url_img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv.warpAffine(img,M,(cols,rows))
    cv.imshow('img',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Rotation
def Rotation():
    img = cv.imread(url_img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
    dst = cv.warpAffine(img,M,(cols,rows))
    cv.imshow('img',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Affine Transformation
def AffineTransform():
    img = cv.imread(url_img)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

# Perspective Transformation

def Perspective():
    img = cv.imread(url_img)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (300, 300))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

def main():
    while 1:
        option = int(input())
        if option == 1: Scale()
        if option == 2: Translation()
        if option == 3: Rotation()
        if option == 4: AffineTransform()
        if option == 5: Perspective()
        if option == 0: break

if __name__ == '__main__':
    main()

