import os
import numpy as np
from keras import models

from config import PROJECT_DIR
from scripts import docImageVectorizer

def is_image_file(file_path):
    """Check if a file is an image

    Args:
        file_path (str): Absolute path of the file
    """
    return file_path.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    )
    
def classify_document(input_data, model_path):
    """Helper function to classify any image doc as resume or not

    Args:
        input_data (str: Takes input from user
        model_path (str): Absolute Path of the Keras model saved

    Returns:
        Str: Classification of the attached document
    """
    if os.path.exists(input_data) and is_image_file(input_data):
        vectorizer = docImageVectorizer(dims=(256, 256))
        img_array = vectorizer.convert_image_to_array(input_data)
        # since model needs batch size
        img_array = np.expand_dims(img_array, axis=0)
        model = models.load_model(model_path)
        prediction = model.predict(img_array)
        return "Resume" if prediction >=0.5 else "Not a resume"
    print("Incompatible file type, please try again!")
        
if __name__ == '__main__':
    input_data = input(
        "Please enter the absolute path of a document saved as an image: \n>>>"
    ).strip()
    MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'resumeModel_best.keras')
    result = classify_document(input_data=input_data, model_path=MODEL_PATH)
    print(result)
