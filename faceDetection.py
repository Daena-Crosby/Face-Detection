import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/daena/OneDrive/Personal Projects/FaceDetect/haarcascade_frontalface_default.xml")

# Open webcam
video_capture = cv2.VideoCapture(0)

# Check if webcam opened
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to improve speed
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display result
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
