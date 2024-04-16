import base64
import json
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from imageio import imread

import learn2learn as l2l

app = Flask(__name__)
cors = CORS(app)

device = 'cpu'

def get_circle(img):
    
    empt_img = np.zeros_like(img)
    
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(img, (3, 3)) 

    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 20, minRadius = 100, maxRadius = 1000) 

    detected_circles = np.uint16(np.around(detected_circles)) 
    pt = detected_circles[0,0]
    a, b, r = pt[0], pt[1], pt[2] 

    # Draw the circumference of the circle. 
    cv2.circle(empt_img, (a, b), r, (255), 20) 

    return empt_img

def crop_border(img, mask=None):
    if mask is not None:
        coords = cv2.findNonZero(mask)
    else:
        coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y + h, x:x + w]
    return rect

def b64_to_tensor(img_b64_str, clock=False):
    image_base64_decode = base64.b64decode(img_b64_str + '=' * (-len(img_b64_str) % 4))
    img = imread(BytesIO(image_base64_decode))
    img = np.squeeze(img[:, :, 3:4], axis=2)

    if clock:
        clock_mask_tmp = get_circle(img)
        clock_mask = cv2.bitwise_and(img, clock_mask_tmp)
        img = cv2.bitwise_xor(img, clock_mask)

    if img.mean() != 0:
        if clock:
            img = crop_border(img, mask=clock_mask)
        else:
            img = crop_border(img)

    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = abs(255 - img)

    # reduce noise in image
    img[img < 100] = 0
    img[img >= 100] = 255
    
    return torch.FloatTensor(img).to(device)

class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class PreProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return b64_to_tensor(x)

class PreProcessClock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return b64_to_tensor(x, clock=True)
    
    
def load_model(task):
    features_path = 'models/adapted_anil_features_{task}_4shots_12steps_64hidden_5layers.pth'.format(task=task)
    head_path = 'models/adapted_anil_head_{task}_4shots_12steps_64hidden_5layers.pth'.format(task=task)
    hidden = 64
    ways = 2

    features = torch.nn.Sequential(l2l.nn.Lambda(lambda x: x.view(-1, 1, 256, 256)),
                            l2l.vision.models.ConvBase(hidden=hidden, channels=1, max_pool=False, layers=5),
                            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
                            Lambda(lambda x: x.view(-1, hidden)))
    features.to(device)
    features.load_state_dict(torch.load(features_path, map_location=device))

    head = torch.nn.Linear(hidden, ways)
    head = l2l.algorithms.MAML(head, lr=0.1)
    head.to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))

    if task == "Clock":
        preprocess = PreProcessClock()
    else:
        preprocess = PreProcess()

    return torch.nn.Sequential(preprocess, features, head)

complex_model = load_model("Complex")
polygon_model = load_model("Polygon")
clock_model   = load_model("Clock")
memory_model  = load_model("Memory")

task_model_dict = {
    "complex": complex_model,
    "polygon": polygon_model,
    "clock": clock_model,
    "memory": memory_model
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello World!"})

@app.route('/predict/<task>', methods=['GET', 'POST'])
def predict(task):
    if request.method == 'GET':
        return jsonify({"error": "Please use POST method"})
    if request.method == 'POST':
        model = task_model_dict[task.lower()]
        if request.data:
            request_data = json.loads(request.data)
            if request_data['image']:
                img_b64_str = request_data['image']
                result_tensor = model(img_b64_str)
                result_int = result_tensor.argmax(dim=1).item()
                result_text = "Dementia" if result_int == 1 else "Healthy"
                return jsonify({"result": "{result_int} - {result_text}".format(result_int=result_int, result_text=result_text)})
            else:
                return jsonify({"error": "No image found"})
        else:
            return jsonify({"error": "No data found"})


if __name__ == '__main__':
    app.run()
