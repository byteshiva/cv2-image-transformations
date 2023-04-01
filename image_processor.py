import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

class ImageProcessor:
    
    def __init__(self, filename):
        self.filename = filename
        self.load_image()
    
    def load_image(self):
        # Load the image
        self.img = cv2.imread(self.filename)
        
    def save_numpy_array(self, filepath):
        # Convert the image to a numpy array
        self.img_array = np.array(self.img)
        # Save the numpy array to disk
        np.save(filepath, self.img_array)
    
    def delete_image(self):
        # Delete the original image
        os.remove(self.filename)
    
    def load_numpy_array(self, filepath):
        # Load the numpy array from disk
        self.img_array = np.load(filepath)
    
    def convert_to_image(self):
        # Convert the numpy array back to an image
        self.img = cv2.cvtColor(self.img_array, cv2.COLOR_RGB2BGR)
    
    def save_image(self, filename):
        # Save the image to disk
        cv2.imwrite(filename, self.img)
    
    def rotate(self, angle):
        # Rotate the image by the specified angle
        self.img = self._rotate_image(self.img, angle)
    
    def _rotate_image(self, image, angle):
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        return cv2.warpAffine(image, M, (cols,rows))
    
    def flip_horizontal(self):
        # Flip the image horizontally
        self.img = np.flip(self.img, axis=1)
    
    def flip_vertical(self):
        # Flip the image vertically
        self.img = np.flip(self.img, axis=0)
    
    def resize(self, width, height):
        # Resize the image to a new width and height
        self.img = cv2.resize(self.img, (width, height))
    
    def blur(self, kernel_size):
        # Blur the image
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)
    
    def show_image(self):
        # Display the image
        cv2_imshow(self.img)


# Create an instance of the ImageProcessor class
img_processor = ImageProcessor('example.jpg')

# Save the image as a numpy array and delete the original image
img_processor.save_numpy_array('example.npy')
# img_processor.delete_image()

# Load the numpy array and convert it back to an image
img_processor.load_numpy_array('example.npy')
img_processor.convert_to_image()

# Create a new instance of the ImageProcessor class
img_processor_rotate = ImageProcessor('example.jpg')
img_processor_rotate.rotate(90)
img_processor_rotate.show_image()

# Create another new instance of the ImageProcessor class
img_processor_flip = ImageProcessor('example.jpg')
img_processor_flip.flip_horizontal()
img_processor_flip.show_image()

# Create yet another new instance of the ImageProcessor class
img_processor_resize = ImageProcessor('example.jpg')
img_processor_resize.resize(500, 500)
img_processor_resize.show_image()

# Create another new instance of the ImageProcessor class
img_processor_blur = ImageProcessor('example.jpg')
img_processor_blur.blur(5)
img_processor_blur.show_image()

# Save the final image
img_processor.save_image('example_processed.jpg')

