import cv2
import numpy as np
import os
from imutils import face_utils
import dlib

# Load the pre-trained face detection model
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def capture_face_details():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)

        for face in faces:
            # Convert face rectangle to (x, y, w, h) format
            (x, y, w, h) = face_utils.rect_to_bb(face)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display frame
            cv2.imshow('Capture Face Details', frame)

            # Ask user for their name
            name = input("Enter your name: ")

            # Detect facial landmarks
            shape = landmark_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Save facial landmarks to a file
            np.save(f"{name}_face.npy", shape)
            print("Face details captured successfully!")

            return name

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = capture_face_details()
