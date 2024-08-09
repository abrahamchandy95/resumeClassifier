import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from keras import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from config import datasets_dir, PROJECT_DIR

if __name__ == '__main__':
    # Load the test datasets
    with open(os.path.join(datasets_dir, 'test_data.pkl')) as f:
        X_test, y_test = pickle.load(f)

    # Load the model
    model_path = os.path.join(PROJECT_DIR, 'models', 'resumeModel_best.keras')
    model = models.load_model(model_path)

    # Calculating y_pred
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype('int32')

    # Print metrics
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1_Score: ', f1_score(y_test, y_pred))

    # Plot confusion matrix
    conf_matrx = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrx, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()