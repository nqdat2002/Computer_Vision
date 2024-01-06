import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

# Define an initial bounding box
bbox = (287, 287, 287, 287)

# Create a CSRT tracker object
tracker = cv2.TrackerCSRT_create()

# Initialize the tracker
tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Draw the bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("CSRT Tracker", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
