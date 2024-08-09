import os
import random
import pickle
from typing import Dict, Tuple


import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, callbacks

from config import datasets_dir, PROJECT_DIR
from scripts import docImageVectorizer


def label_images(
    label_files:Dict[str, str], image_dir: str
)-> Dict[str, int]:
    """
    Returns a dictionary of image file paths and the associated 
    label. 1 for resume and 0 for everything else.

    Args:
        label_files (Dict[str, str]): a dictionary of the text files 
        taken from the documentation of the dataset used along with 
        their absolute paths. These files must be opened by our
        program
        image_dir (str): Absolute directory of where all the images 
        were downloaded
    """
    labeled_images = {}
    for _, filepath in label_files.items():
        with open(filepath, 'r') as f:
            for line in f:
                rel_path, label = line.strip().split(' ')
                full_path = os.path.join(image_dir, rel_path)
                labeled_images[full_path] = 1 if label=='14' else 0
    return labeled_images

def balance_labels(
    labeled_images:Dict[str, int]
)-> Dict[str, int]:
    """
    Ensures that the labeled dictionary is balanced by sampling 
    the number of non-resumes to match the number of resumes.

    Args:
        labeled_images (Dict[str, int]): Dictionary of image paths 
        with their corresponding label. There are many more of one 
        label than the other.

    Returns:
        Dict[str, int]: A Dictionary of image paths with their label.
        Here, the number of images classified as resumes will match 
        that of the non-resumes to ensure balanced data.
    """
    resumes = []
    non_resumes = []
    for path, label in labeled_images.items():
        if label == 1:
            resumes.append(path)
        else:
            non_resumes.append(path)
    non_resumes = random.sample(non_resumes, len(resumes))
    # Create a new balanced dictionary
    bal_labeled_images = {path:1 for path in resumes}
    bal_labeled_images.update({path: 0 for path in non_resumes})
    return bal_labeled_images

def extract_features_and_labels(
    labeled_images:Dict[str, int], vectorizer:docImageVectorizer
)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features and labels from the given dictionary
    and returns test and train datasets

    Args:
        labeled_images (Dict[str, int]): A dictionary of absolute paths
        with their labels
        vectorizer (docImageVectorizer): An instance of the custom class
        to made to vectorize images of documents

    Returns:
        Features and labels to be used for a machine learning model
        with data type compatability
        - X_train (np.ndarray): Features for the training dataset.
        - X_test (np.ndarray): Features for the test dataset.
        - y_train (np.ndarray): Labels for the training dataset.
        - y_test (np.ndarray): Labels for the test dataset.
    """
    image_paths = list(labeled_images.keys())
    labels = list(labeled_images.values())
    # Extract features
    image_features = vectorizer.extract_features_from_images(
        image_paths=image_paths
    )
    # Split data into test and training datasets
    X_train, X_test, y_train, y_test = train_test_split(
        image_features, labels, test_size=0.2, random_state=42
    )
    # Convert labels to a compatible data type
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    return X_train, X_test, y_train, y_test

class ResumeClassifierCNNModel:
    """
    A CNN model that Classifies an image file as a resume or non-resume
    
    Attributes:
        image_shape (tuple): The shape of the input images with
        format (height, width, channels). Number of channels is 1
        for this assignment.
        checkpoint_dir (str): Absolute path of the directory where the
        model will be saved.
        model (Keras Model): The compiled neural network model 
    
    Methods:
    _build_model(): Constructs the CNN architecture
    _compile_model(): Compiles the model with the specified optimizer
    get_model(): Returns the compiled model    
    """
    def __init__(self, image_shape, checkpoint_dir):
        self.image_shape = image_shape
        self.checkpoint_dir = checkpoint_dir
        self.model = self._build_model()
        self._compile_model()
        
    def _build_model(self):
        """
        Builds a convolutional neural network using 
        Keras' Sequential model.
        
        Returns:
            Sequential: A Keras Sequential model with CNN 
            architecture for binary classification.
        """
        model = models.Sequential([
            layers.Input(shape=self.image_shape),
            
            layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.2),
            
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.3),
            
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(1, activation='sigmoid')
        ])
        return model


    def _compile_model(self):
        """
        Compiles the model with the optimizer and learning rate
        """
        optimizer = optimizers.Adam(learning_rate=0.00005)
        self.model.compile(
            optimizer=optimizer, loss='binary_crossentropy', 
            metrics=['accuracy']
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(
            self.checkpoint_dir, 'resumeModel_best.keras'
        )

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss', patience=10, verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def get_model(self):
        """
        Returns the compiled Keras model.
        """
        return self.model


if __name__ == "__main__":
    
    img_dir = os.path.join(datasets_dir, 'images')
    label_files = {
        'train': os.path.join(datasets_dir, 'labels', 'train.txt'),
        'test': os.path.join(datasets_dir, 'labels', 'test.txt'),
        'val': os.path.join(datasets_dir, 'labels', 'val.txt')
    }
    # Create dictionary of image_path and label
    labeled_images = label_images(label_files=label_files, image_dir=img_dir)
    # Make the labels balanced: number of resumes = non-resumes
    labeled_imgs = balance_labels(labeled_images=labeled_images)
    # Extract features and labels from images
    dims = (256, 256)
    vectorizer = docImageVectorizer(dims=dims)
    X_train, X_test, y_train, y_test = extract_features_and_labels(
        labeled_images=labeled_imgs, vectorizer=vectorizer
    )
    # Train the model
    batch_size = 32
    epochs=20
    image_shape = (dims[0], dims[1], 1)
    checkpoint_dir = os.path.join(PROJECT_DIR, 'models')
    model = ResumeClassifierCNNModel(
        image_shape=image_shape, checkpoint_dir=checkpoint_dir
    )
    model = model.get_model()
    model.fit(
        X_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=model.callbacks
    )
    # Save the test datasets for future evaluation
    with open(os.path.join(datasets_dir, 'test_data.pkl')) as f:
        pickle.dump((X_test, y_test), f)