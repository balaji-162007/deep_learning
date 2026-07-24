import face_recognition
import cv2
import os
import numpy as np

# Folder containing known faces
path = "known_faces"

known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(path):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        image = face_recognition.load_image_file(
            os.path.join(path, filename)
        )

        # Generate CNN face encoding
        encoding = face_recognition.face_encodings(
            image,
            model="cnn"
        )

        if len(encoding) > 0:
            known_face_encodings.append(encoding[0])
            known_face_names.append(
                os.path.splitext(filename)[0]
            )

print("Known faces loaded:", known_face_names)


# Open webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces using CNN
    face_locations = face_recognition.face_locations(
        rgb_frame,
        model="cnn"
    )

    # Extract face encodings
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for face_encoding, face_location in zip(
            face_encodings,
            face_locations):

        # Compare with database
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=0.5
        )

        name = "Unknown"

        # Find closest match
        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)

            if matches[best_match]:
                name = known_face_names[best_match]

        # Draw rectangle
        top, right, bottom, left = face_location

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "CNN Face Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
