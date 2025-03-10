from PIL import Image
import matplotlib.pyplot as plt
import os
import random #to make sure get some images from the different sub files
from torchvision.transforms import v2
import random
import torch
 
dataset_folder = "Animal Image Dataset" #The dataset that contains all the images

 
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
 
outputs=[]
inputs=[]
for animal_folder in animal_folders:
    animal_folder_path = os.path.join(dataset_folder, animal_folder)
    image_files = os.listdir(animal_folder_path)
    num_images_to_process = min(3, len(image_files))  # Process up to 3 images per folder
    
    for image_file in image_files:
        image_path = os.path.join(animal_folder_path, image_file)
        
        picture = Image.open(image_path)
        picture = resize_transforms(picture)
        picture = augment_transforms(picture)
        
        outputs.append(animal_folder)
        inputs.append (picture)
        # Convert the tensor back to a numpy array for visualization
        picture = picture.permute(1, 2, 0).numpy()
        picture = picture * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        picture = picture.clip(0, 1)
         
        plt.imshow(picture)
        plt.title(f"Augmented Image: {image_file}")
        plt.show()
if inputs:   
    inputs = torch.stack(inputs)  
    print(f"Processed {len(inputs)} images. Final tensor shape: {inputs.shape}")
 


 