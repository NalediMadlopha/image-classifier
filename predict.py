import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models

import json
import os
from PIL import Image

from utils import *

data_transforms =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
]) 

args = prediction_args()
model = None

def main():
    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.input, model, args.top_k)

    if args.category_names == None:
        print("There is a {:.3f} probability that this flower falls under category {}".format(max(probs), classes[probs.index(max(probs))]))
    else:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        flower_names = get_class_names(cat_to_name, classes, model.class_to_idx)        
        most_likely_flower = flower_names[probs.index(max(probs))]
        
        print("There is a {:.3f} probability that this is a(n) {}".format(max(probs), most_likely_flower.title()))
  
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(models, checkpoint['arch'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.features = checkpoint['features']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['idx_to_class']

    return model


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Predicts the class from an image file
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = process_image(image_path)
    image_unsqueeze = image.unsqueeze(0).to(device)

    prob = torch.exp(model.forward(image_unsqueeze))
    probs, classes = prob.topk(topk)

    probs = probs.detach().cpu().numpy().tolist()[0]
    classes = classes.detach().cpu().numpy().tolist()[0]

    return probs, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Processes a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transform = data_transforms
    
    return transform(img)


def get_class_names(cat_to_name, classes, class_to_idx):
    names = {}
    for class_key in class_to_idx:
        names[class_to_idx[class_key]] = cat_to_name[class_key]
        
    return [names[class_key] for class_key in classes]


if __name__ == "__main__":
    main()