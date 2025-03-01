from PIL import Image
import matplotlib.pyplot as plt
import os
import random #to make sure get some images from the different sub files
from torchvision.transforms import v2
import random
 
dataset_folder = "CMPM-17-FirstRepo/archive/Animal Image Dataset" #The dataset that contains all the images

 
animal_folders = os.listdir(dataset_folder)

resize_transforms = v2,Compose([
    v2.ToTensor(),
    v2.Resize([225,300]),
])

augment_transforms - v2.Compose([
    v2.RandomResizedCrop(size=(225, 300),  antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
random_animal = []

for animal_category in animal_folders:
    augmented_animals = 0
 
    while augmented_animals<(len(os.listdir(animal_category)/2)):
        random_animal.append(random.choice(os.listdir(animal_category)))
        augmented_animals += 1

transforms = v2.Compose()
        
