import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (255,255, 0), -1)
cv2.circle(img, (250, 100), 50, (0, 255, 255), -1)
cv2.ellipse(img, (300, 100), (80, 40), 0, 0, 360, (255, 0, 255), -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    M = cv2.moments(contour)
    area = M['m00']
    perimeter = cv2.arcLength(contour, True)
    roundness = (perimeter ** 2) / (4 * np.pi * area)
    if len(contour) >= 5:
        _, (a, b), _ = cv2.fitEllipse(contour)
        ecc = np.sqrt(1 - int(b / a) ** 2)
    else:
        ecc = 1

    x, y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    cv2.putText(img, f'Roundness: {roundness:.2f}', (x - 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f'Eccentricity: {ecc:.2f}', (x - 70, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()