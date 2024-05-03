from fastapi import FastAPI
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from torchvision import models,transforms,datasets
import os
import shutil
import time
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import PIL.Image
from IPython.display import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchvision
# Define the FastAPI app
app = FastAPI()

class_dict = {0 : "safe driving",
              1 : "texting - right",
              2 : "talking on the phone - right",
              3 : "texting - left",
              4 : "talking on the phone - left",
              5 : "operating the radio",
              6 : "drinking",
              7 : "reaching behind",
              8 : "hair and makeup",
              9 : "talking to passenger"}
transform = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor()])

@app.post("/test/") 
def test():
    return "Test başarılı olarak gerçekleşti."


@app.post("/predict_yolov8/") # url YoloV8
def predict_yolov8(data_id : int):

    
    data = f"./images/img_{data_id}.jpg"

    load_yolo = YOLO('models/best_yoloV8l.pt')
    prediction = load_yolo.predict(data)
    
    predicted_class = prediction[0].probs.top1
    print(predicted_class)

    return {"predicted_class": predicted_class}

@app.post("/predict_ResNet50/") # url YoloV8
def predict_ResNet50(data_id : int):

    
        
    data = f"./images/img_{data_id}.jpg"
    
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load("./models/resnet50_driver_state", map_location=torch.device('cpu')))
    model.eval()
    im_path = data
    with PIL.Image.open(im_path) as im:
        im = transform(im)
        im = im.unsqueeze(0)
        output = model(im.to('cpu'))
        proba = nn.Softmax(dim=1)(output)
        proba = [round(float(elem),4) for elem in proba[0]]
        print(proba)
        print("Predicted class:",class_dict[proba.index(max(proba))])
        print("Confidence:",max(proba))
        proba2 = proba.copy()
        proba2[proba2.index(max(proba2))] = 0.
        print("2nd answer:",class_dict[proba2.index(max(proba2))])
        print("Confidence:",max(proba2))
    predicted_class = class_dict[proba.index(max(proba))]
    confidence = max(proba)
    
    proba2 = proba.copy()
    proba2[proba2.index(max(proba2))] = 0.
    second_class = class_dict[proba2.index(max(proba2))]
    second_confidence = max(proba2)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "second_class": second_class,
        "second_confidence": second_confidence
    }




@app.get("/")
async def read_root():
    return {"message": "Welcome to the Driver State Prediction API. Created by Ömer Faruk Ballı"}
    


# Run the FastAPI app
# python -m uvicorn ann:app --reload