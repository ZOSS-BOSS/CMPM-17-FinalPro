import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        self.num_classes = len(self.animal_folders)
        
        # Load images and labels
        for folder in self.animal_folders:
            folder_path = os.path.join(root_dir, folder)
            for img_file in os.listdir(folder_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder_path, img_file))
                    self.labels.append(folder)
        
        # Convert labels to one-hot encoding
        self.labels = pd.DataFrame(self.labels)
        self.labels = pd.get_dummies(self.labels, columns=[0])
        self.labels = torch.tensor(self.labels.values, dtype=torch.float)
        
        print(f"Loaded {len(self.image_paths)} images across {len(self.animal_folders)} classes")
        print(f"One-hot encoded labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx, :]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations with basic augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transform for visualization
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
    vis_loader = DataLoader(vis_dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(vis_loader))
    
    plt.figure(figsize=(12, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        label_idx = torch.argmax(labels[i]).item()
        class_name = vis_dataset.idx_to_class[label_idx]
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
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = dataset.num_classes
model = AnimalCNN(num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Helper function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    true_labels = torch.argmax(labels, dim=1)
    correct = (predicted == true_labels).sum().item()
    total = labels.size(0)
    return correct, total

# Training function with accuracy tracking
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            batch_correct, batch_total = calculate_accuracy(outputs, labels)
            correct += batch_correct
            total += batch_total
            
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Test accuracy for this epoch
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")
    
    # Plot the metrics
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, 'g-o', label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accuracies, 'r-o', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return train_accuracies[-1], test_accuracies[-1]

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(labels, dim=1)
            
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Test function with per-class accuracy
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(labels, dim=1)
            
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(true_labels)):
                label = true_labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Print overall accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {dataset.idx_to_class[i]}: {class_accuracy:.2f}%")
    
    return accuracy

# Display some sample images
print("Displaying sample images from the dataset:")
show_sample_images()

# Train the model
print("\nTraining the model...")
final_train_acc, final_test_acc = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5)

# Test the model
print("\nTesting the model...")
test_accuracy = test_model(model, test_loader)

# Print final results
print("\nFinal Results:")
print(f"Final Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), 'animal_cnn_model.pth')
print("Model saved as 'animal_cnn_model.pth'")

# Function to show model predictions
def show_predictions():
    test_loader_small = DataLoader(test_dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(test_loader_small))
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels = torch.argmax(labels, dim=1)
    
    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        true_class = dataset.idx_to_class[true_labels[i].item()]
        pred_class = dataset.idx_to_class[predicted[i].item()]
        
        plt.imshow(img)
        color = "green" if true_class == pred_class else "red"
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Show some predictions
print("\nShowing some predictions:")
show_predictions()