import cv2
import numpy as np
import matplotlib.pyplot as plt

url_img ='CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg'
img = cv2.imread(url_img)
mask_square = np.zeros_like(img)
cv2.rectangle(mask_square, (100, 100), (300, 300), (255, 255, 255), -1)
mask_circle = np.zeros_like(img)

cv2.circle(mask_circle, (400, 200), 100, (255, 255, 255), -1)
result_and_square = cv2.bitwise_and(img, mask_square)
result_or_circle = cv2.bitwise_or(img, mask_circle)
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Ảnh Gốc')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask_square, cmap='gray')
plt.title('Mặt nạ Hình Vuông')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(result_and_square)
plt.title('Kết Quả AND')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(mask_circle, cmap='gray')
plt.title('Mặt nạ Hình Tròn')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(result_or_circle)
plt.title('Kết Quả OR')
plt.axis('off')

plt.show()
