import cv2
from typing import Dict
import face_recognition
from google.colab.patches import cv2_imshow
import numpy as np
from typing import Dict, List, Tuple

class FaceRecognizer:
    def __init__(self, known_faces: Dict[str, face_recognition.face_encodings], unknown_color: Tuple[int,int,int]=(0,0,255)):
        self.known_faces = known_faces
        self.unknown_color = unknown_color
    
    def label_faces(self, image: np.ndarray) -> np.ndarray:
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find all the faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Label the faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Initialize a variable to store the name of the person
            name = "Unknown"

            # Compare the face encoding with all known faces
            for known_name, known_face_encoding in self.known_faces.items():
                if face_recognition.compare_faces([known_face_encoding], face_encoding)[0]:
                    # If a match is found, set the name variable
                    name = known_name

            # Determine the color to use for the face label
            color = self.unknown_color if name == "Unknown" else (0, 255, 0)  # default to red for unknown faces

            # Draw a rectangle around the face and label it
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return image

    def recognize_faces(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        labeled_image = self.label_faces(image)
        cv2_imshow(labeled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
known_faces = {
    "Person 1": face_recognition.face_encodings(face_recognition.load_image_file("person1.png"))[0],
    "Person 2": face_recognition.face_encodings(face_recognition.load_image_file("person2.png"))[0]
}

face_recognizer = FaceRecognizer(known_faces, unknown_color=(0, 0, 128))  # specify blue for unknown faces
face_recognizer.recognize_faces("peoples.jpg")
