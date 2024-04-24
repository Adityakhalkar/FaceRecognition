import cv2
import numpy as np
import os
from imutils import face_utils
import dlib

# Load the pre-trained face detection model
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def recognize_face():
    # Load person details from the files
    known_faces = {}
    for filename in os.listdir("."):
        if filename.endswith("_face.npy"):
            name = os.path.splitext(filename)[0].split("_")[0]
            known_faces[name] = np.load(filename)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)

        for face in faces:
            # Convert face rectangle to (x, y, w, h) format
            (x, y, w, h) = face_utils.rect_to_bb(face)

            # Detect facial landmarks
            shape = landmark_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Compare detected face with known faces
            authenticated = False
            for name, known_face in known_faces.items():
                mse = np.mean((known_face - shape) ** 2)
                if mse < 5000:  # Adjust MSE threshold as needed
                    # Label the face with the person's name
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    authenticated = True
                    print("Authenticated as", name)
                    break

            if authenticated:
                break  # Break out of the loop if face is authenticated

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or authenticated:
            break  # Break out of the loop if 'q' is pressed or face is authenticated

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
