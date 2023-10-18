import cv2

def handlerLeftMouseClickRect(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(image_with_legend, "B20DCPT053", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        cv2.rectangle(image_with_legend, (x - 10, y + 10), (x + 100, y - 30), (255, 0, 0), thickness=2)
        cv2.imshow("Image with Legend", image_with_legend)

def handlerLeftMouseClickCircle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(image_with_legend, "B20DCPT053", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        cv2.circle(image_with_legend, (x, y), 30, (0, 255, 0), thickness=2)
        cv2.imshow("Image with Legend", image_with_legend)

url_img = "CVFALL2023B20DCPT053/CVFALL2023B20DCPT053002.jpg"
img = cv2.imread(url_img)
image_with_legend = img

cv2.namedWindow("Image with Legend")
cv2.setMouseCallback("Image with Legend", handlerLeftMouseClickRect)

while True:
    cv2.imshow("Image with Legend", img)

    if cv2.waitKey(0) == ord('s'):
        newimg_path = url_img.replace(".jpg", f"wlegend.jpg")
        cv2.imwrite(newimg_path, image_with_legend)
        break

cv2.destroyAllWindows()