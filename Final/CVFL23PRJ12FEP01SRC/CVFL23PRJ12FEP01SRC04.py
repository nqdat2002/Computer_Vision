import cv2

# Load the video
cap = cv2.VideoCapture('video.mp4')

# Create the FAST object detector
fast = cv2.FastFeatureDetector_create()

# Loop through the frames of the video
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the features using the FAST algorithm
    kp = fast.detect(gray, None)

    # Draw the detected features on the frame
    img = cv2.drawKeypoints(frame, kp, None, color=(0,255,0))

    # Display the resulting frame
    cv2.imshow('frame',img)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
