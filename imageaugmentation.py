from PIL import Image
import matplotlib.pyplot as plt
import os
import random #to make sure get some images from the different sub files
from torchvision.transforms import v2
 
dataset_folder = "CMPM-17-FirstRepo/archive/Animal Image Dataset" #The dataset that contains all the images

 
animal_folders = os.listdir(dataset_folder)

random_animal = []

for animal_category in animal_folders:
    augmented_animals = 0
 
    while augmented_animals<100:
        random_animal.append(random.choice(os.listdir(animal_category)))
        augmented_animals += 1
        
