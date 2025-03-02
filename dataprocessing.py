from PIL import Image
import matplotlib.pyplot as plt
import os
import random #to make sure get some images from the different sub files
from torchvision.transforms import v2
from torch.utils.data import Dataset
dataset_folder = "Animal Image Dataset" #The dataset that contains all the images
delete_img = ("CMPM-17-FinalPro/Animal Image Dataset/butterfly/e134b50a2efc1c22d2524518b7444f92e37fe5d404b0144390f8c770a1edb6_640.jpg")
# Check if the image exists and delete it
if os.path.exists(delete_img):
    os.remove(delete_img)
animal_folders = os.listdir(dataset_folder) #creating a list for the subfolders in our dataset folder

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize([225,])
])
image_files = [] # creating a filepath to store the images 
 

for animal_folder in animal_folders:#I am making a loop to collect the images
    animal_folder_path = os.path.join(dataset_folder, animal_folder) #join paths without error
    
    if os.path.isdir(animal_folder_path):# Make sure it's a folder (not a file)
        
        for image_file in os.listdir(animal_folder_path):# grabbing the image from the subfolder 
            if image_file.endswith(('.jpg')):# Check if the file is an image  
                image_files.append(os.path.join(animal_folder_path, image_file)) #putting the image in this list

 
images_to_display = min(100, len(image_files)) #Setting the amount of images to 100 for now
random.shuffle(image_files)# ensuring they randomly choose images from all the sub folders
 
plt.figure(figsize=(15, 15))#use this to adjust the window and can put in multiple images


 
for idx, imagepath in enumerate(image_files[:images_to_display]):#created a loop to loop through the images
     
    img = Image.open(imagepath)#using PIL to open and display the images
    
    # using matplotlib
    plt.subplot(15, 15, idx +1)  # 15x15 grid for up to 100 images
    plt.imshow(img)
    plt.axis("off")  # Hiding the axes
   

plt.tight_layout() #found this on matplotlib to make sure no overlay betwwen images
plt.show()#plotting
