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


class AnimalDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.animal_folders = os.listdir(dataset_folder)

        # Dynamically calculate the number of classes
        self.num_classes = len(self.animal_folders)  # The number of unique folders will be the number of classes

        for label, folder in enumerate(self.animal_folders):
            folder_path = os.path.join(dataset_folder, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg')):
                        self.image_paths.append(os.path.join(folder_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")  # RGB format

        
        if self.transform:
            image = self.transform(image)

        # One-hot encode the label
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1

        return image, one_hot_label


class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 75, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 75)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


dataset_folder = "Animal Image Dataset"  # Replace with the actual path to your dataset folder

dataset = AnimalDataset(dataset_folder, transform=transform)

# Split the dataset into training and testing (50% training, 50% testing)
train_size = int(0.50 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = AnimalCNN(num_classes=dataset.num_classes)  # Dynamically set the number of classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_loop(dataloader):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.argmax(dim=1))  # CrossEntropy expects the index of the class, not one-hot
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Train Batch {batch_idx + 1}: Loss = {loss.item()}")


def test_loop(dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target.argmax(dim=1)).sum().item()  # Compare with one-hot encoded label
            total += target.size(0)
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")


print("Training Loop:")
train_loop(train_loader)

print("\nTesting Loop:")
test_loop(test_loader)

def forward():
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()