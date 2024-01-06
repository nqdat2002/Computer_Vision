import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

# Create an ORB tracker object
tracker = cv2.ORB_create()

# Initialize the tracker
tracker.setMaxFeatures(500)
tracker.setFastThreshold(0)

kp = tracker.detect(frame, None)
kp, des = tracker.compute(frame, kp)

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Update the tracker
    kp = tracker.detect(frame, None)
    kp, des = tracker.compute(frame, kp)

    # Draw the keypoints
    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    # Display the resulting frame
    cv2.imshow("ORB Tracker", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
