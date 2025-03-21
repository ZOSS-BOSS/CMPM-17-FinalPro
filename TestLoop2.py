import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os
import pandas as pd
import wandb

# Create a dataframe of path and label
dataset_folder = "Animal Image Dataset" #The dataset that contains all the images
animal_folders = os.listdir(dataset_folder)
image_paths = []

for animal_type in animal_folders:
    for animal_image_path in os.listdir(dataset_folder + "/" + animal_type):
        image_paths.append([dataset_folder + "/" + animal_type + "/" + animal_image_path,animal_type])

image_paths = pd.DataFrame(image_paths)
image_paths = pd.get_dummies(image_paths, columns = [1])
# print(image_paths.loc[2:5])


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
        print(self.values.head)
        input = self.values.iloc[idx,0]
        output = self.values.iloc[idx,1:12]
        input = Image.open(input).convert("RGB")
        input = transform(input)  # RGB format
        return input, output


training_data = image_paths.loc[0:int((len(image_paths)*0.7))]
testing_data = image_paths.loc[int((len(image_paths)*0.7)):int(len(image_paths)*0.85)]
validation_data = image_paths.loc[int((len(image_paths)*0.85)):]

train_dataset = AnimalDataset(training_data)
test_dataset = AnimalDataset(testing_data)
validation_dataset = AnimalDataset(validation_data)
#full dataset
print(train_dataset)
# Split the dataset into training and testing (50% training, 50% testing)
 
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#Create dataloaders
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
validate_loader = DataLoader(validation_dataset, batch_size = 8, shuffle = True)

loss_func = nn.CrossEntropyLoss()

class ConvModel(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32,16, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d (kernel_size = 2, stride = 2)
        self.lin1 = nn.Linear(int((225*300*16)/64),12)
    
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

conv_model = ConvModel()
optimizer = torch.optim.Adam(conv_model.parameters(),lr = 0.01, weight_decay=0.01)
run = wandb.init(project="Image Convolution", name = "first run")

for input, output in train_loader:
    for vinputs, voutputs in validate_loader:
        pred = conv_model(input)
        loss = loss_func(pred, output)
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






    import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class for Animal Image Dataset
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get animal folders (classes)
        self.animal_folders = []
        for folder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder)):
                self.animal_folders.append(folder)
        
        # Create class to index mapping
        self.class_to_idx = {folder: idx for idx, folder in enumerate(self.animal_folders)}
        self.idx_to_class = {idx: folder for folder, idx in self.class_to_idx.items()}
        
        # Load images and labels
        for folder in self.animal_folders:
            folder_path = os.path.join(root_dir, folder)
            for img_file in os.listdir(folder_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder_path, img_file))
                    self.labels.append(self.class_to_idx[folder])
        
        print(f"Loaded {len(self.image_paths)} images across {len(self.animal_folders)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations with basic augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a smaller size for faster training
    transforms.RandomHorizontalFlip(),  # Basic data augmentation
    transforms.RandomRotation(10),  # Small random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transform for visualization (no augmentation)
simple_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
dataset = AnimalDataset("Animal Image Dataset", transform=transform)
vis_dataset = AnimalDataset("Animal Image Dataset", transform=simple_transform)

# Split into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to display sample images
def show_sample_images():
    # Create a small dataloader for visualization
    vis_loader = DataLoader(vis_dataset, batch_size=5, shuffle=True)
    
    # Get a batch of images
    images, labels = next(iter(vis_loader))
    
    # Create a figure
    plt.figure(figsize=(12, 3))
    
    # Display each image
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        
        # Convert tensor to numpy image (for visualization)
        img = images[i].permute(1, 2, 0).numpy()
        
        # Get the class name
        label_idx = labels[i].item()
        class_name = vis_dataset.idx_to_class[label_idx]
        
        # Display the image
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Define a simple CNN model
class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Simple dropout to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(dataset.class_to_idx)
model = AnimalCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    # Lists to track metrics
    train_losses = []
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Testing function
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    
    # Tracking variables
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    # No gradient calculation needed
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Print overall accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print per-class accuracy
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {dataset.idx_to_class[i]}: {class_accuracy:.2f}%")
    
    return accuracy

# Display some sample images with labels
print("Displaying sample images from the dataset:")
show_sample_images()

# Train the model
print("\nTraining the model...")
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Test the model
print("\nTesting the model...")
test_model(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'animal_cnn_model.pth')
print("Model saved as 'animal_cnn_model.pth'")

# Function to show model predictions
def show_predictions():
    # Load a few test images
    test_loader_small = DataLoader(test_dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(test_loader_small))
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Display images with predictions
    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        
        # Convert to numpy for display
        img = images[i].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get true and predicted class names
        true_class = dataset.idx_to_class[labels[i].item()]
        pred_class = dataset.idx_to_class[predicted[i].item()]
        
        # Display image with labels
        plt.imshow(img)
        color = "green" if true_class == pred_class else "red"
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Show some predictions
print("\nShowing some predictions:")
show_predictions()