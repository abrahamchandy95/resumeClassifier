import numpy as np
from typing import List, Tuple
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

class docImageVectorizer:
    def __init__(self, dims: Tuple[int, int]) -> None:
        self.dims = dims
    
    def convert_image_to_array(
        self, image_path: str
    ) -> np.ndarray:
        """Converts an image into a numpy array representing features

        Args:
            image_path (str): Absolute path of the image

        Returns:
            np.ndarray: The output array with the shape of 
            (height, width, 1)
        """
        image_array = imread(image_path)
        if len(image_array.shape) == 3:
            image_array = rgb2gray(image_array)
        image_array = resize(
            image_array, (self.dims[0], self.dims[1]), anti_aliasing=True
        )
        if image_array.ndim == 2:
            # Add dimension at the end
            image_array = np.expand_dims(image_array, axis=-1)
        return image_array.astype(np.float32)
    
    def extract_features_from_images(
        self, image_paths: List[str]
    ) -> np.ndarray:
        """From a list of images, this function extracts an array 
        representation of the images

        Args:
            image_paths (List[str]): List of each image for conversion

        Returns:
            np.ndarray: The output array with shape of 
            (len(image_paths), height, width, 1).
        """
        num_images = len(image_paths)
        # initialize with zeros
        features = np.zeros(
            (num_images, self.dims[0], self.dims[1], 1),
            dtype=np.float32
        )
        for idx, img_path in enumerate(image_paths):
            features[idx] = self.convert_image_to_array(img_path)
        return features