import io

import flask
from flask_cors import CORS
import torch
import base64
import numpy as np

import PIL.Image
from torch import nn
from torchvision import transforms


# load pytorch model
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)

# replace the last layer of resnet18 with a Sequential layer (lin, relu, drop, lin)
resnet18.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
resnet18.load_state_dict(torch.load('resnet18.pth', map_location=torch.device('cpu')))
resnet18.eval()
# create flask app
app = flask.Flask(__name__)
CORS(app)


# create endpoint
def base64_to_image(param):
    print(param)
    img = base64.urlsafe_b64decode(param)
    return img

img_dims=(224, 224)
img_transforms = transforms.Compose([
    transforms.Resize(img_dims),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    # /predict endpoint will be used to make predictions based on an image,
    # which will be sent as base 64 encoded string
    # returns "Cat" or "Dog"

    # print all request data
    img_form_param=flask.request.files["img"]

    img = PIL.Image.open(img_form_param)

    shape=np.array(img).shape
    if shape[2]==4:
        img = img.convert('RGB')

    # convert image to PIL image
    # transform image
    #pil_img= PIL.Image.open(io.BytesIO(img))
    img = img_transforms(img)
    # predict with pytorch model

    result=resnet18(img.unsqueeze(0))
    if result[0].argmax()==0:
        return 'Cat  -  ' + str(result[0])
    else:
        return 'Dog  -  ' + str(result[0])

# start flask app on port 8081
app.run(host='localhost', port=8081)
#%%
