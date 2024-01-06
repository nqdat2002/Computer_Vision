import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Linear Transform
def linear_transform():
    img = cv.imread('./data/images/anh1.jpg')
    assert img is not None, "File could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    scale_factor_x = 2
    scale_factor_y = 2
    M = np.array([[scale_factor_x, 0, 0], [0, scale_factor_y, 0]], dtype=np.float32)
    dst = cv.warpAffine(img, M, (cols, rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Rotation
def rotation():
    img = cv.imread('./data/images/anh1.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols = img.shape
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
    dst = cv.warpAffine(img,M,(cols,rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Shear
def shear():
    img = cv.imread('./data/images/anh1.jpg')
    assert img is not None, "File could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    shear_factor = 0.2
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    dst = cv.warpAffine(img, M, (cols, rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Translation
def translation():
    img = cv.imread('./data/images/anh1.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv.warpAffine(img,M,(cols,rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Similarity transformation
def similarity_transformation():
    img = cv.imread('./data/images/anh1.jpg')
    assert img is not None, "File could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    scale_factor = 1.5
    rotation_angle = 45
    translation_x = 50
    translation_y = -30
    center = (cols // 2, rows // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, scale_factor)
    translation_matrix = np.array([[1, 0, translation_x], [0, 1, translation_y]], dtype=np.float32)
    similarity_matrix = np.dot(rotation_matrix, np.vstack([translation_matrix, [0, 0, 1]]))
    dst = cv.warpAffine(img, similarity_matrix[:2, :], (cols, rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Affine Transformation
def affine_transformation():
    img = cv.imread('./data/images/anh1.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(img,M,(cols,rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

# Projective
def projective():
    img = cv.imread('data/images/anh1.jpg')
    assert img is not None, "File could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    enlarged_width = cols + 750
    enlarged_height = rows + 750
    pts2 = np.float32([[0, 0], [enlarged_width, 0], [0, enlarged_height], [enlarged_width, enlarged_height]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    cmap = 'gray'
    plt.subplot(121), plt.imshow(img, cmap=cmap), plt.title('Input')
    plt.subplot(122), plt.imshow(dst, cmap=cmap), plt.title('Output')
    plt.show()

if __name__ == '__main__':
    while 1:
        print("0: Kết thúc\n1: Linear_transform\n2: Rotation\n3: Shear\n4: Translation\n5: Similarity Tranformation\n6: Affine Transformation\n7: Projective\n")
        option = int(input("Nhập lựa chọn của bạn: "))
        if option == 0:
            print("Kết thúc chương trình\n")
            break
        if option == 1:
            linear_transform()
        if option == 2:
            rotation()
        if option == 3:
            shear()
        if option == 4:
            translation()
        if option == 5:
            similarity_transformation()
        if option == 6:
            affine_transformation()
        if option == 7:
            projective()