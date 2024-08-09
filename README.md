# Resume Classifier

## Overview
This project is designed to classify images of documents as either resumes or non-resumes.
It used the RVL-CDIP dataset to train a convolutional neural network (CNN) model.
You can automatically download the dataset by running the following command on your terminal

python src/scripts/datasetsDownloader.py

This scipt first creates an empty directory named 'datasets in your project's root folder, 
then all the necessary files are extracted from the downloaded .tar file. Among these extracted
files, there is documentation that is essential for the labeling process used later in the project.

## Model Training

To train the CNN model, execute the following script in the project's root directory:

python src/processing/resume_classifier.py

This one file contains code to preprocess, vectorize and manages the training loop for the project.

## Model Evaluation

Evaluate the model's performance after training using the script below:

python/src/eval/model_evaluation.py

This script, when run plots the confusion matrix, and we get measures of accuracy, precision, recall and f1 score.

## Making a prediction

The main.py is designed to interact with the user. To get this model to classify an image as a resume or not, please run:

python src/main.py

Here, you will be prompted to enter the absolute path of an image, 
and the model should correctly classify the image as a resume!
