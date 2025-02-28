import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
import torch

categories = []
folderpath = "C:/Users/Zane/Documents/CPMP-17-ML/CMPM-17-FirstRepo/archive/Animal Image Dataset/"
for path in os.listdir(folderpath):
    # print(path)
    categories.append(folderpath + path + "/")

transforms = v2.Compose (
    [v2.ToTensor()]
)
for i in range(25):
    animal = random.choice(categories)
    picture = os.listdir(animal)
    picture = animal + random.choice(picture)
    picture = Image.open(picture)
    
    picture = transforms(picture)
    
    print(picture.shape)




    