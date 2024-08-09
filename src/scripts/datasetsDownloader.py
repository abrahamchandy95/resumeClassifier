import os
import requests
import tarfile

from config import datasets_dir

"""RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset"""
_CITATION = """\
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}
"""

_DESCRIPTION = """\
The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images.
"""

_HOMEPAGE = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"

_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_URL = {
    "rvl-cdip": "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz",
}

_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

if __name__ == "__main__":
    # Ensure the datasets_dir exists
    os.makedirs(datasets_dir, exist_ok=True)
    # Downloading the raw file
    RVL_URL = _URL["rvl-cdip"]
    TARFILE_PATH = os.path.join(datasets_dir, RVL_URL.split('/')[-1])
    if not os.path.exists(TARFILE_PATH):
        print("Tar file does not exist. Downloading..")
        with requests.get(RVL_URL, stream=True) as r:
            r.raise_for_status()
            with open(TARFILE_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        #extract from .tar file
        with tarfile.open(TARFILE_PATH, "r:gz") as tar:
            tar.extractall(path=datasets_dir)
    else:
        print("Tar file exists")
