import cv2
import numpy as np
import matplotlib.pyplot as plt
url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"

img = cv2.imread(url_img, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_image = cv2.dilate(img, kernel, iterations=1)
eroded_image = cv2.erode(img, kernel, iterations=1)
opened_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh đa mức xám')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opened_image, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closed_image, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.show()

img = cv2.imread(url_img, cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Ảnh nhị phân')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opened_image, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closed_image, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.show()


