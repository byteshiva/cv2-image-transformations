import cv2
import face_recognition
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image


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

    def recognize_faces(self, image_path: str, output_path: str, max_width: int = 800, max_height: int = 800) -> None:
        # Load the image
        image = cv2.imread(image_path)

        # Reduce the image size if it exceeds the max width or height
        if image.shape[0] > max_height or image.shape[1] > max_width:
            scale_factor = min(max_width/image.shape[1], max_height/image.shape[0])
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height))

        # Label the faces in the image
        labeled_image = self.label_faces(image)

        # Save the labeled image
        labeled_image = Image.fromarray(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
        labeled_image.save(output_path)


# Example usage
IMG_DIR = "./img/"
IMG_SRC_PERSON1 = IMG_DIR + "person1.png"
IMG_SRC_PERSON2 = IMG_DIR + "person2.png"
IMG_SRC_PEOPLES = IMG_DIR + "peoples.jpg"
IMG_LABElED_PEOPLES = IMG_DIR + "labeled_peoples.jpg"

known_faces = {
    "Person 1": face_recognition.face_encodings(face_recognition.load_image_file(IMG_SRC_PERSON1))[0],
    "Person 2": face_recognition.face_encodings(face_recognition.load_image_file(IMG_SRC_PERSON2))[0]
}

face_recognizer = FaceRecognizer(known_faces, unknown_color=(0, 0, 128))  #
