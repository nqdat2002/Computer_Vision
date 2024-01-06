import cv2
import numpy as np
import matplotlib.pyplot as plt

# Can bang luoc do tan suat cho anh da muc xam

def equalize_frequency_histogram():

    img = cv2.imread('./data/images/anh1.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "File could not be read, check with os.path.exists()"
    # Áp dụng cân bằng histogram
    equ = cv2.equalizeHist(img)
    # Hiển thị ảnh gốc và ảnh sau khi cân bằng histogram
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(equ, cmap='gray'), plt.title('Equalized Image')
    plt.show()
    # Hiển thị lược đồ tần suất của ảnh gốc và ảnh sau khi cân bằng histogram
    plt.figure(figsize=(12, 4))
    plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(142), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Original Histogram')

    plt.subplot(143), plt.imshow(equ, cmap='gray'), plt.title('Equalized Image')
    plt.subplot(144), plt.hist(equ.ravel(), 256, [0, 256]), plt.title('Equalized Histogram')

    plt.show()

# Can bang luoc do tan suat cho anh mau

def equalize_frequency_histogram_for_color():

    img = cv2.imread('./data/images/anh1.jpg')
    assert img is not None, "File could not be read, check with os.path.exists()"
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Áp dụng cân bằng histogram cho kênh giá trị (Value) trong không gian màu HSV
    hsv_img[:,:,2] = cv2.equalizeHist(hsv_img[:,:,2])

    # Chuyển đổi ảnh trở lại từ HSV sang BGR
    equalized_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # Hiển thị ảnh gốc và ảnh sau khi cân bằng histogram
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)), plt.title('Equalized Image')
    plt.show()

    # Hiển thị lược đồ tần suất của kênh giá trị (Value) trước và sau khi cân bằng histogram
    plt.figure(figsize=(12, 4))

    plt.subplot(141), plt.imshow(hsv_img[:,:,2], cmap='gray'), plt.title('Value Channel (Equalized)')
    plt.subplot(142), plt.hist(hsv_img[:,:,2].ravel(), 256, [0, 256]), plt.title('Equalized Value Histogram')
    plt.subplot(143), plt.hist(img[:,:,2].ravel(), 256, [0, 256]), plt.title('Original Value Histogram')
    plt.subplot(144), plt.imshow(img[:,:,2], cmap='gray'), plt.title('Value Channel (Original)')
    plt.show()

# Can bang luoc do tan suat thich nghi (Adaptive Histogram Equalization)
def equalize_frequency_histogram_adaptive():
    img = cv2.imread('data/images/anh1.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "File could not be read, check with os.path.exists()"

    # Áp dụng cân bằng lược đồ tần suất thích nghi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng cân bằng lược đồ tần suất thích nghi
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img_clahe, cmap='gray'), plt.title('CLAHE Image')
    plt.show()

    # Hiển thị lược đồ tần suất của ảnh gốc và ảnh sau khi áp dụng cân bằng lược đồ tần suất thích nghi
    plt.figure(figsize=(12, 4))

    plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(142), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Original Histogram')

    plt.subplot(143), plt.imshow(img_clahe, cmap='gray'), plt.title('CLAHE Image')
    plt.subplot(144), plt.hist(img_clahe.ravel(), 256, [0, 256]), plt.title('CLAHE Histogram')
    plt.show()

if __name__ == '__main__':
    while 1:
        print("0: Kết thúc\n1: equalize_frequency_histogram\n2: equalize_frequency_histogram_for_color\n3: equalize_frequency_histogram_adaptive\n")
        option = int(input("Nhập lựa chọn của bạn: "))
        if option == 0:
            print("Kết thúc chương trình\n")
            break
        if option == 1:
            equalize_frequency_histogram()
        if option == 2:
            equalize_frequency_histogram_for_color()
        if option == 3:
            equalize_frequency_histogram_adaptive()