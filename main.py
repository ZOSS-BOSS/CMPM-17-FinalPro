import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import os
import wandb
import pandas as pd
from torchvision.transforms import v2


# Initialize WandB for tracking the run
wandb.init(project="Animal Image Classification", name="animal-cnn-run")

transform = transforms.Compose([  # transformations for data augmentation and normalization
    transforms.RandomResizedCrop(size=(225, 300), scale=(0.5, 1), antialias=True),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AnimalDataset(Dataset):  # Dataset class
    def __init__(self, dataset_folder, transform=None):
        self.transform = transform
        self.image_paths = []  # List to store all the images in the dataset
        self.labels = []
        self.animal_folders = os.listdir(dataset_folder)  # Subfolders in the dataset

        # Get the number of classes by the number of subfolders
        self.classes = sorted(self.animal_folders)  # Sort the folders alphabetically
        self.num_classes = len(self.classes)  # Number of unique animal classes

        # Iterate over each folder to add images and labels
        for label, folder in enumerate(self.classes):
            folder_path = os.path.join(dataset_folder, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                      # You can also add other formats like .png if needed
                        self.image_paths.append(os.path.join(folder_path, file))
                        self.labels.append(folder)  # Label corresponds to the folder index
        # print(self.labels)
        self.labels = pd.DataFrame(self.labels)
        self.labels = pd.get_dummies(self.labels, columns = [0])
        # print(self.labels.shape)
        self.labels = torch.tensor(self.labels.values, dtype = torch.int)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx, :]
        # print({"label:"}, label)
        image = Image.open(image_path).convert("RGB")  # Convert to RGB format

        if self.transform:
            image = self.transform(image)

        return image, label  # Return the image and its class label


class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # MaxPooling layer

        self.fc1 = nn.Linear(64 * 56 * 75, 512)  # Adjust this size later if needed
        self.fc2 = nn.Linear(512, 12)  # Output layer should have a number of units equal to num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 56 * 75)  # Flattening the output for fully connected layers

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer with the number of classes
        return x


dataset_folder = "Animal Image Dataset"  # Path to the dataset

# Instantiate the dataset
dataset = AnimalDataset(dataset_folder, transform=transform)

# Split the dataset into training and testing sets (80-20 split)
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, loss function, and optimizer setup
model = AnimalCNN(num_classes=dataset.num_classes)  # Dynamically set the number of classes
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


# Training loop function with accuracy calculation
def train_loop(dataloader):
    model.train()  # Set the model to training mode
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)  # Forward pass
        print(output.shape)
        # print(target.shape)
        loss = criterion(output, target)  # CrossEntropyLoss expects class labels as integers
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        
        # Calculate accuracy during training
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()  # Compare predictions with actual labels
        total += target.size(0)

        if batch_idx % 10 == 0:  # Print loss and accuracy every 10 batches
            accuracy = 100 * correct / total
            print(f"Train Batch {batch_idx + 1}: Loss = {loss.item()}, Accuracy = {accuracy:.2f}%")
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})  # Log to WandB

# Testing loop function with accuracy calculation
def test_loop(dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during testing for efficiency
        for data, target in dataloader:
            output = model(data)  # Forward pass
            _, predicted = torch.max(output, 1)  # Get the predicted class
            correct += (predicted == target).sum().item()  # Compare with actual label
            total += target.size(0)
        
        accuracy = 100 * correct / total  # Calculate accuracy
        print(f"Test Accuracy: {accuracy:.2f}%")
        wandb.log({"test_accuracy": accuracy})  # Log test accuracy to WandB


# Train and Test the Model
print("Training Loop:")
train_loop(train_loader)  # Running the training loop

print("\nTesting Loop:")
test_loop(test_loader)  # Running the testing loop