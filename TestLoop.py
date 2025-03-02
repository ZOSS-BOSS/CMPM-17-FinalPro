import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

  
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(225, 300), scale=(0.5, 1), antialias=True),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # from the image_augumentation.py
])

 
class AnimalDataset(Dataset):#Create a class that inherits from the PyTorch Datasets

    def __init__(self, dataset_folder, transform=None):
        self.transform = transform# to apply image transformations
        self.image_paths = []
        self.labels = []
        self.animal_folders = os.listdir(dataset_folder)#listing the folders in the dataset

        for label, folder in enumerate(self.animal_folders):
            folder_path = os.path.join(dataset_folder, folder)
            if os.path.isdir(folder_path):  # Ensure it's a folder, not a file
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg')):   
                        self.image_paths.append(os.path.join(folder_path, file))# adding the image to the list
                        self.labels.append(label)#adding the animal types to the list

    def __len__(self):#to return the # of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")  # RGB format

        
        if self.transform:#applying transformations
            image = self.transform(image)

        return image, label

dataset_folder = "Animal Image Dataset"  # Replace with the actual path to your dataset folder

 
dataset = AnimalDataset(dataset_folder, transform=transform)#full dataset

# Split the dataset into training and testing (50% training, 50% testing)
train_size = int(0.50 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

 
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#Create dataloaders
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

 
def train_loop(dataloader):#train loop
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Train Batch {batch_idx + 1}:")
        print("Input Data (Image Tensor):", data.shape)  # Print the shape of the image tensor
        print("Target Labels:", target)


def test_loop(dataloader):#test loop
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Test Batch {batch_idx + 1}:")
        print("Input Data (Image Tensor):", data.shape)  # Print the shape of the image tensor
        print("Target Labels:", target)

 
print("Training Loop:")
train_loop(train_loader)

print("\nTesting Loop:")
test_loop(test_loader)
