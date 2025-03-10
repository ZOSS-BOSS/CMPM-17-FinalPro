from PIL import Image
import matplotlib.pyplot as plt
import os
import random #to make sure get some images from the different sub files
from torchvision.transforms import v2
import random
 
dataset_folder = "CMPM-17-FirstRepo/archive/Animal Image Dataset" #The dataset that contains all the images

 
animal_folders = os.listdir(dataset_folder)

resize_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size = (225,300)),
])

augment_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(225, 300), scale = (0.5, 1), antialias=True),
    v2.RandomHorizontalFlip(p=0.3),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
])

picture  = Image.open("CMPM-17-FirstRepo/archive/Animal Image Dataset/cow/OIP-_Dx1fsxBCBSXQ_kgAXAxVwHaDP.jpeg")
picture = resize_transforms(picture)
plt.imshow(picture)
plt.show()


        
