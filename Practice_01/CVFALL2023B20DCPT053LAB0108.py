import cv2
import numpy as np
import matplotlib.pyplot as plt
url_img = 'CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg'
image = cv2.imread(url_img, cv2.IMREAD_GRAYSCALE)

canny_edges = cv2.Canny(image, 100, 200)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

hx_mask = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
hy_mask = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

sobel_x_hx = cv2.filter2D(image, -1, hx_mask)
sobel_y_hy = cv2.filter2D(image, -1, hy_mask)

gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title('Kết quả Canny')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(sobel_x, cmap='gray')
plt.title('Đạo hàm theo X (Sobel)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(sobel_y, cmap='gray')
plt.title('Đạo hàm theo Y (Sobel)')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(sobel_x_hx, cmap='gray')
plt.title('Đạo hàm theo X (Hx)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(sobel_y_hy, cmap='gray')
plt.title('Đạo hàm theo Y (Hy)')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Mô-đun Véc-tơ Gradient')
plt.axis('off')

plt.show()
