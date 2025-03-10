import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import wandb

# Create a dataframe of path and label
dataset_folder = "C:/Users/zspeiser/Desktop/code/CMPM-17-FinalPro/Animal Image Dataset/" #The dataset that contains all the images
animal_folders = os.listdir(dataset_folder)
image_paths = []

for animal_type in animal_folders:
    for animal_image_path in animal_type:
        image_paths.append([dataset_folder + "/" + animal_type + "/" + animal_image_path,animal_type])

image_paths = pd.DataFrame(image_paths)

print(image_paths)
print(image_paths[0])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=(225, 300), scale=(0.5, 1), antialias=True),
    transforms.RandomHorizontalFlip(p=0.3),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # from the image_augumentation.py
])

 
class AnimalDataset(Dataset):#Create a class that inherits from the PyTorch Datasets

    def __init__(self, input):
       self.values = input
       #listing the folders in the dataset

    def __len__(self):#to return the # of images in the dataset
        return len(self.values)

    def __getitem__(self, idx):
        print(idx)
        input = self.values[idx,0]
        output = self.values[idx,1]
        input = Image.open(input).convert("RGB")
        input = transform(input)  # RGB format
        return input, output


training_data = image_paths[0:int((len(image_paths)*0.7))]
testing_data = image_paths[int((len(image_paths)*0.7)):int(len(image_paths)*0.85)]
validation_data = image_paths[int((len(image_paths)*0.85)):]
 
train_dataset = AnimalDataset(training_data)
test_dataset = AnimalDataset(testing_data)
validation_dataset = AnimalDataset(validation_data)
#full dataset

# Split the dataset into training and testing (50% training, 50% testing)
 
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#Create dataloaders
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
validate_loader = DataLoader(validation_dataset, batch_size = 8, shuffle = True)
 
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

loss_func = nn.CrossEntropyLoss()

class ConvModel(nn.Module):
   
    def __init__(self):
        self.conv1 = nn.Conv(3,32,kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv(32,32,kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv(32,16, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d (kernel_size = 2, stride = 2)
        self.lin1 = nn.Linear((225*300*16)/64,12)
    
    def forward(self, input):
        partial = self.conv1(input)
        partial = self.relu(partial)
        partial = self.maxpool(partial)
        partial = self.conv2(partial)
        partial = self.relu (partial)
        partial = self.maxpool (partial)
        partial = self.conv3 (partial)
        partial = self.relu (partial)
        partial = self.maxpool (partial)
        partial = partial.flatten(start_dim = 1)
        partial = self.lin1(partial)
        return partial

optimizer = torch.optim.Adam(myHouseModel.parameters(),lr = 0.01, weight_decay=0.01)
conv_model = ConvModel()
run = wandb.init(project="Image Convolution", name = "first run")

for inputs, outputs in train_loader:
    for vinputs, voutputs in validate_loader:
        pred = conv_model(inputs)
        loss = loss_func(pred, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            val_pred = ConvModel(vinputs)
            val_loss = loss_func(val_pred, voutputs)
            print (val_loss)
        run.log({"train loss": loss, "validation loss":val_loss})
        
for inputs, outputs in test_loader:
    pred = conv_model(inputs)
    loss = loss_func(pred, outputs)
    run.log({"test loss":loss})