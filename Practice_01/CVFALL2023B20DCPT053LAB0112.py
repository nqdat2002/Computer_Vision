import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

w, h = 1000, 600
img = np.zeros((h, w, 3), dtype=np.uint8)

cv2.circle(img, (300, 300), 50, (0, 0, 255), -1)
cv2.rectangle(img, (400, 100), (450, 250), (0, 255, 0), -1)
pts = np.array([[200, 400], [50, 400], [150, 400]], np.int32)
cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=5)
cv2.ellipse(img, (600, 400), (100, 50), 0, 0, 360, (255, 255, 0), -1)
pentagon_pts = np.array([[600, 500], [750, 600], [150, 600], [650, 550], [700, 450]], np.int32)
cv2.polylines(img, [pentagon_pts], isClosed=True, color=(0, 255, 255), thickness=2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.drawContours(img, [contour], -1, color, -1)

cv2.imshow("Shapes", img)
cv2.imwrite('CVFALL2023B20DCPT053LAB011201.jpg', img)
cv2.waitKey(0)

cv2.destroyAllWindows()