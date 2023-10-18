import cv2
import numpy as np
import matplotlib.pyplot as plt
url_img ='CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg'
image = cv2.imread(url_img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0, 0), 3)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
laplace = cv2.Laplacian(gray, cv2.CV_64F)
laplace_abs = cv2.convertScaleAbs(laplace)
laplace_custom1 = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
laplace_custom2 = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32))

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_abs_x = cv2.convertScaleAbs(sobel_x)
sobel_abs_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

# plt
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(sharp, cmap='gray')
plt.title('Unsharp Masking')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(laplace_abs, cmap='gray')
plt.title('Bộ lọc Laplace')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(laplace_custom1, cmap='gray')
plt.title('Bộ lọc Tương Tương Sự 1')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(laplace_custom2, cmap='gray')
plt.title('Bộ lọc Tương Tương Sự 2')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Bộ lọc Sobel')
plt.axis('off')

plt.show()
