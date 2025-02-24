import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Webcam Test', frame)

    # Break loop if 'q' is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Webcam Test', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
