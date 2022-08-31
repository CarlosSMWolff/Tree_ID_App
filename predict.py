from fastai.vision.all import *
import timm
#import sys
from PIL import Image
import numpy as np


# Load model
learn = load_learner("export_ConvNext_Trees_and_Bushes.pkl")
categories = learn.dls.vocab


def classify_image(im_file_path):
    # Load image from input path
    im= PILImage.create(im_file_path)

    pred, idx, probs = learn.predict(im)
    return dict(zip(categories, map(float,probs)))


# Example:

im_file_path = "leaf1.jpg"
prediction = classify_image(im_file_path)
probs = np.array([prediction[category] for category in categories])
order = np.argsort(-probs)
print("Prediction")
print("========")
print(f'1st: {categories[order[0]]} ({100*probs[order[0]]:.2f}%)')
print(f'2nd: {categories[order[1]]} ({100*probs[order[1]]:.2f}%)')
